import os
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def get_profile(user_id: str):
    return (
        supabase.table("profiles")
        .select("*")
        .eq("user_id", user_id)
        .single()
        .execute()
        .data
    )

def list_vectorstores_for_org(org_id: str):
    return (
        supabase.table("vectorstores")
        .select("id,name,openai_vector_store_id")
        .eq("org_id", org_id)
        .execute()
        .data
    )
