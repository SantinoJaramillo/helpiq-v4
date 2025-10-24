# backend/app/supa.py
import os
from typing import List, Dict, Any
from supabase import create_client, Client

_sb: Client | None = None

def _get_sb() -> Client:
    global _sb
    if _sb is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # backend: service role key
        _sb = create_client(url, key)
    return _sb

def list_vectorstores_for_org(org_id: str) -> List[Dict[str, Any]]:
    sb = _get_sb()
    res = (
        sb.table("vectorstores")
          .select("id,name,openai_vector_store_id,created_at")
          .eq("org_id", org_id)
          .order("created_at", desc=True)
          .execute()
    )
    return res.data or []
