import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from chatkit.store import SQLiteStore  # enkel dev-store (byt till Postgres i produktion)
from chatkit.server import StreamingResult
from starlette.responses import StreamingResponse, JSONResponse

from .server import ServiceTechChatServer
from .supa import list_vectorstores_for_org

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# En enkel “store” för ChatKit-trådar (byt till egen DB om du vill)
data_store = SQLiteStore(path=":memory:")  # byt till fil eller Postgres i prod
server = ServiceTechChatServer(data_store, attachment_store=None)

@app.get("/health")
def health():
    return {"ok": True}

# Lista modeller (vector stores) - hårdkodat org-id just nu för enkelhet
@app.get("/api/models")
def get_models(org_id: str):
    rows = list_vectorstores_for_org(org_id)
    return {"models": rows}

# ChatKit endpoint (ett enda)
@app.post("/chatkit")
async def chatkit_endpoint(request: Request, mode: str = "web", vs: str | None = None):
    body = await request.body()
    # Skicka "context" till ChatKitServer → AgentContext (vi läser i respond())
    result = await server.process(body, {"mode": mode, "vs": vs})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    else:
        return JSONResponse(content=result.json)
