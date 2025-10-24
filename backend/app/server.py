import os
from typing import Any, AsyncIterator

from fastapi import Request
from agents import Agent, Runner, FileSearchTool, WebSearchTool
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.agents import AgentContext, stream_agent_response, simple_to_agent_input
from chatkit.types import ThreadMetadata, UserMessageItem, ThreadStreamEvent

CHATKIT_DOMAIN_KEY = os.environ["CHATKIT_DOMAIN_KEY"]

# En enkel "server" som ChatKit pratar med.
class ServiceTechChatServer(ChatKitServer):
    def __init__(self, store, attachment_store=None):
        super().__init__(store, attachment_store)

    # Bas-instruktioner som gäller båda lägena
    BASE_INSTRUCTIONS = (
        "Du hjälper servicetekniker att felsöka. "
        "Förklara steg-för-steg, och föreslå säkra kontroller först. "
        "Var konkret och citera källsnuttar när det går."
    )

    async def respond(
        self,
        thread: ThreadMetadata,
        input: UserMessageItem | None,
        context: Any,   # här stoppar vi in mode/vector_store_id från requesten
    ) -> AsyncIterator[ThreadStreamEvent]:

        # Läs läge & vector store från request-context (satt i main.py)
        mode = (context or {}).get("mode")          # "manual" eller "web"
        vs_id = (context or {}).get("vs")           # openai_vector_store_id

        # Bygg agenten olika beroende på läge
        if mode == "manual" and vs_id:
            agent = Agent[AgentContext](
                name="Manualsök",
                model="gpt-4.1-mini",
                instructions=self.BASE_INSTRUCTIONS + " Använd endast dokumentationen i vald modell.",
                tools=[FileSearchTool(vector_store_ids=[vs_id], max_num_results=4)],
            )
        else:
            # default = webbsök
            agent = Agent[AgentContext](
                name="Webbsök",
                model="gpt-4.1-mini",
                instructions=self.BASE_INSTRUCTIONS + " Använd webbsök för att hitta svar.",
                tools=[WebSearchTool()],
            )

        # Kör agenten & streama till ChatKit UI
        ctx = AgentContext(thread=thread, store=self.store, request_context=context)
        run = Runner.run_streamed(agent, await simple_to_agent_input(input) if input else [], context=ctx)

        async for ev in stream_agent_response(ctx, run):
            yield ev
