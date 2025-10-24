# backend/app/main.py
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import logging
import traceback

from openai import OpenAI

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from app.supa import list_vectorstores_for_org

# ChatKit
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.types import (
    Attachment,
    Page,
    ThreadItem,
    ThreadMetadata,
    ThreadStreamEvent,
)
from chatkit.store import Store, AttachmentStore

# ChatKit helpers för Agents SDK
from chatkit.agents import ThreadItemConverter, stream_agent_response

# OpenAI Agents SDK (importvägen är "from agents import Agent, Runner")
from agents import Agent, Runner

logger = logging.getLogger("helpiq")
logging.basicConfig(level=logging.INFO)


# ---------- Minimal in-memory store (för utveckling) ----------

class MemoryAttachmentStore(AttachmentStore[dict]):
    def __init__(self) -> None:
        self._attachments: Dict[str, Attachment] = {}

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        self._attachments.pop(attachment_id, None)

    async def create_attachment(self, input, context: dict) -> Attachment:
        # Tvåstegsupload stöds inte här. Vi sätter bara metadata.
        att_id = self.generate_attachment_id(input.mime_type, context)
        att = Attachment(id=att_id, type=input.type, name=input.name, mime_type=input.mime_type)
        self._attachments[att_id] = att
        return att

    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        att = self._attachments.get(attachment_id)
        if not att:
            raise KeyError(f"Attachment {attachment_id} not found")
        return att


class MemoryStore(Store[dict]):
    """
    Väldigt enkel in-memory store. Bra nog för Render-deploy & test.
    I produktion: ersätt med en Postgres/Supabase-implementation.
    """
    def __init__(self) -> None:
        self._threads: Dict[str, ThreadMetadata] = {}
        self._items: Dict[str, List[ThreadItem]] = {}

    # --- Threads & items ---
    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        return self._threads[thread_id]

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        self._threads[thread.id] = thread
        self._items.setdefault(thread.id, [])

    async def load_threads(self, limit: int, after: Optional[str], order: str, context: dict) -> Page[ThreadMetadata]:
        threads = list(self._threads.values())
        items = threads[: limit if limit else len(threads)]
        return Page[ThreadMetadata](items=items, next=None)

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        self._items.setdefault(thread_id, []).append(item)

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        items = self._items.setdefault(thread_id, [])
        for i, it in enumerate(items):
            if it.id == item.id:
                items[i] = item
                break
        else:
            items.append(item)

    async def load_item(self, thread_id: str, item_id: str, context: dict) -> ThreadItem:
        for it in self._items.get(thread_id, []):
            if it.id == item_id:
                return it
        raise KeyError(f"Item {item_id} not found in thread {thread_id}")

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        self._threads.pop(thread_id, None)
        self._items.pop(thread_id, None)

    async def delete_thread_item(self, thread_id: str, item_id: str, context: dict) -> None:
        items = self._items.get(thread_id, [])
        self._items[thread_id] = [it for it in items if it.id != item_id]

    async def load_thread_items(self, thread_id: str, after: Optional[str], limit: int, order: str, context: dict) -> Page[ThreadItem]:
        items = self._items.get(thread_id, [])
        slice_items = items[: limit if limit else len(items)]
        return Page[ThreadItem](items=slice_items, next=None)

    # --- Attachments (inte använda här – no-ops för att uppfylla Store-gränssnittet) ---
    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        return None

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        raise NotImplementedError("Use a separate AttachmentStore in this demo")

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        return None


# ---------- ChatKit Server som använder OpenAI Agents SDK ----------

converter = ThreadItemConverter()

class MyChatKitServer(ChatKitServer[dict]):
    def __init__(self, store: Store[dict], attachment_store: Optional[AttachmentStore[dict]] = None):
        super().__init__(store=store, attachment_store=attachment_store)

    # ⬇️ Sätt modell explicit
    assistant_agent = Agent(
        name="Assistant",
        instructions="You are a helpful troubleshooting assistant for service technicians. Answer clearly and concisely.",
        model="gpt-4o-mini",   # <— välj en lätt modell
    )

    async def respond(
        self,
        thread: ThreadMetadata,
        item: Optional["ThreadItem"],
        context: dict,
    ) -> AsyncIterator[ThreadStreamEvent]:
        try:
            # ⬇️ faila tidigt om API-nyckeln saknas
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set on the server")

            # kör agenten streamat
            result_stream = await Runner.run_streamed(
                self.assistant_agent, converter, thread
            )

            async for event in stream_agent_response(result_stream, converter, thread, item):
                yield event

        except Exception as e:
            logger.exception("Agent streaming failed")
            # låt ChatKit hantera som stream.error (du såg den tidigare)
            raise


# ---------- FastAPI app, CORS & endpoints ----------

app = FastAPI(title="HelpIQ ChatKit backend")

# ✅ CORS – tillåt helpiq.se + localhost (Vite) + explicit headers för preflight
origins = [
    os.getenv("ALLOWED_ORIGIN", "http://localhost:5173"),
    "http://localhost:5173",
    "http://localhost:3000",
    "https://helpiq.se",
    "https://www.helpiq.se",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                     # ingen "*" ihop med credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # räcker
    allow_headers=[                            # 👈 explicit är nyckeln här
        "Content-Type",
        "X-OpenAI-Domain-Key",
        "Authorization",
    ],
    expose_headers=["Content-Type"],
    max_age=86400,
)

# ✅ Health & Root (enkla att testa)
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True}

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

@app.head("/health", include_in_schema=False)
def health_head():
    return Response(status_code=200)

@app.get("/debug/diag", include_in_schema=False)
def diag():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    try:
        import agents  # noqa
        agents_ok = True
    except Exception:
        agents_ok = False
    return {"has_openai_api_key": has_key, "agents_pkg_ok": agents_ok}

@app.get("/debug/openai", include_in_schema=False)
def debug_openai():
    # enkel icke-streamad ping mot OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        return {"ok": False, "why": "OPENAI_API_KEY missing"}
    try:
        client = OpenAI()
        r = client.responses.create(model="gpt-4o-mini", input="Skriv ordet: PONG")
        # hämta första text-output
        txt = ""
        for out in r.output or []:
            if out.type == "output_text":
                txt += out.text
        return {"ok": True, "text": txt}
    except Exception as e:
        return {"ok": False, "why": str(e)}


# --- Models API (för SelectModel) ---
@app.get("/api/models")
def list_models(org_id: str):
    """
    Returnerar vectorstores för en given org_id som:
    { "models": [ { id, name, openai_vector_store_id }, ... ] }
    """
    try:
        rows = list_vectorstores_for_org(org_id)
        models = [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "openai_vector_store_id": r.get("openai_vector_store_id") or r.get("id"),
            }
            for r in (rows or [])
        ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Initiera store/server
data_store = MemoryStore()
attachment_store = MemoryAttachmentStore()
server = MyChatKitServer(data_store, attachment_store)

@app.options("/chatkit", include_in_schema=False)
async def chatkit_options():
    return Response(status_code=200)


# ✅ Preflight (OPTIONS) finns redan ovan

@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    """
    ChatKit skickar alla anrop (JSON eller SSE).
    Vi låter ChatKitServer processa och returnerar antingen JSON eller SSE-stream.
    """
    body = await request.body()

    # Skicka vidare query-params till servern som context
    mode = request.query_params.get("mode")
    vs = request.query_params.get("vs")
    context = {"mode": mode, "vs": vs}

    # --- CORS headers (explicit) ---
    origin = request.headers.get("origin")
    allowed = {
        os.getenv("ALLOWED_ORIGIN", "http://localhost:5173"),
        "http://localhost:5173",
        "http://localhost:3000",
        "https://helpiq.se",
        "https://www.helpiq.se",
    }
    cors_headers = {}
    if origin in allowed:
        cors_headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": "Content-Type",
            "Cache-Control": "no-cache",
        }

    try:
        result = await server.process(body, context=context)

        if isinstance(result, StreamingResult):
            return StreamingResponse(result, media_type="text/event-stream", headers=cors_headers)

        return Response(content=result.json, media_type="application/json", headers=cors_headers)

    except Exception as e:
        # Viktigt: returnera CORS-headers även när det går fel
        # (och logga felet i Render-loggarna)
        err = {"error": "internal_error", "message": str(e)}
        return Response(
            content=str(err),
            media_type="application/json",
            headers=cors_headers,
            status_code=500,
        )


# Render kör: uvicorn app.main:app --host 0.0.0.0 --port $PORT
