# backend/app/main.py
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

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

    # En väldigt enkel agent som svarar hjälpsamt
    assistant_agent = Agent(
        name="Assistant",
        instructions="You are a helpful troubleshooting assistant for service technicians. Answer clearly and concisely.",
        # valfri: model="gpt-4o-mini"
    )

    async def respond(
        self,
        thread: ThreadMetadata,
        item: Optional["ThreadItem"],  # UserMessageItem när användaren skriver
        context: dict
    ) -> AsyncIterator[ThreadStreamEvent]:
        """
        Kör agenten strömmat och konvertera till ChatKit events.
        """
        # (Valfri logik: använd context.get("mode") / context.get("vs"))
        result_stream = await Runner.run_streamed(
            self.assistant_agent,
            converter,   # converter kan hantera tråd -> agent-input
            thread,      # hela ThreadMetadata
        )

        async for event in stream_agent_response(result_stream, converter, thread, item):
            yield event


# ---------- FastAPI app, CORS & endpoints ----------

app = FastAPI(title="HelpIQ ChatKit backend")

# ✅ CORS – tillåt helpiq.se + localhost (Vite)
origins = [
    os.getenv("ALLOWED_ORIGIN", "http://localhost:5173"),  # kan sättas i Render
    "http://localhost:5173",
    "http://localhost:3000",
    "https://helpiq.se",
    "https://www.helpiq.se",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # inte "*" ihop med allow_credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# --- Models API (för SelectModel) ---
@app.get("/api/models")
def list_models(org_id: str):
    """
    Returnerar vectorstores för en given org_id som:
    { "models": [ { id, name, openai_vector_store_id }, ... ] }
    """
    # Lazy-import så att appen inte kraschar vid uppstart om env saknas.
    try:
        try:
            from .supa import list_vectorstores_for_org  # om supa.py ligger i samma paket
        except ImportError:
            from supa import list_vectorstores_for_org   # fallback om den ligger bredvid
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase ej konfigurerat: {e}")

    try:
        rows = list_vectorstores_for_org(org_id)
        # Säkerställ fältens namn som frontenden väntar sig
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

# ✅ Preflight (OPTIONS) för /chatkit
@app.options("/chatkit", include_in_schema=False)
async def chatkit_options():
    # Låt CORSMiddleware svara med 200 + rätt headers
    return Response(status_code=200)

@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    """
    ChatKit skickar alla anrop (JSON eller SSE).
    Vi låter ChatKitServer processa och returnerar antingen JSON eller SSE-stream.
    """
    body = await request.body()

    # ⬇️ Skicka vidare query-params till servern som context
    mode = request.query_params.get("mode")  # "manual" | "web"
    vs = request.query_params.get("vs")      # openai vector store id
    context = {"mode": mode, "vs": vs}

    result = await server.process(body, context=context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")


# Render kör: uvicorn app.main:app --host 0.0.0.0 --port $PORT
