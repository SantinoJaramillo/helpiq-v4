import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Dev-store som följer med openai-chatkit (ingen extra modul krävs)
from chatkit.server import StreamingResult
from chatkit.sqlite import SQLiteStore

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

# 💾 Spara på disk så att trådar överlever om processen startas om
os.makedirs("data", exist_ok=True)
data_store = SQLiteStore(path="data/chatkit.db")

server = ServiceTechChatServer(data_store, attachment_store=None)

@app.get("/health")
def health():
    return {"ok": True}

# Lista vectorstores (modeller) för ett org-id
@app.get("/api/models")
def get_models(org_id: str):
    rows = list_vectorstores_for_org(org_id)
    return {"models": rows}

# ChatKit-endpoint (JSON/SSE)
@app.post("/chatkit")
async def chatkit_endpoint(request: Request, mode: str = "web", vs: str | None = None):
    body = await request.body()
    result = await server.process(body, {"mode": mode, "vs": vs})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return JSONResponse(content=result.json)
