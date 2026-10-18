# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.genai import types
from pydantic import BaseModel, Field

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.observability import (
    get_recent_exported_spans,
    get_recent_telemetry_events,
    redact_pii,
)
from app.tools import TOOL_JSON_SCHEMAS

load_dotenv()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
otel_to_cloud = True

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


class TravelChatRequest(BaseModel):
    """Request schema for the interactive Captain Layover Web UI chat endpoint."""

    message: str = Field(..., min_length=1, description="Traveler message text.")
    session_id: str = Field(
        default="web-default-session", description="Conversation session identifier."
    )
    user_id: str = Field(
        default="web_traveler", description="Traveler user identifier."
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "funny-travel-agent"
app.description = "API for interacting with the Agent funny-travel-agent"

# Remove default root redirect so `/` serves the Captain Layover Travel Agency Web UI
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/travel-ui", response_class=HTMLResponse, include_in_schema=False)
async def serve_travel_web_ui() -> HTMLResponse:
    """Serve the interactive Captain Layover Travel Agency Web UI."""
    if STATIC_INDEX_HTML.exists():
        return HTMLResponse(content=STATIC_INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Captain Layover Travel Agency UI</h1>")


@app.post("/api/travel/chat")
async def travel_chat_endpoint(req: TravelChatRequest) -> dict[str, Any]:
    """Execute a conversational turn through the ADK Runner for the Web UI."""
    runner: Runner = app.state.runner
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=req.message)],
    )
    reply_parts: list[str] = []
    tool_calls_made: list[str] = []

    async for event in runner.run_async(
        user_id=req.user_id,
        session_id=req.session_id,
        new_message=content,
    ):
        ev_content = getattr(event, "content", None)
        if ev_content and getattr(ev_content, "parts", None):
            for part in ev_content.parts:
                if getattr(part, "text", None):
                    reply_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    tool_calls_made.append(fc.name)

    session = await runner.session_service.get_session(
        app_name=app.state.agent_app_name,
        user_id=req.user_id,
        session_id=req.session_id,
    )
    state = getattr(session, "state", {}) if session else {}
    pending_hitl = state.get("pending_hitl_booking")
    if "user_confirmed=True" in req.message or "CONFIRMED" in req.message.upper():
        pending_hitl = None

    return {
        "status": "success",
        "session_id": req.session_id,
        "reply": redact_pii(
            "\n".join(reply_parts).strip() or "Captain Layover standing by!"
        ),
        "tool_calls": tool_calls_made,
        "routed_model": state.get("last_routed_model", "gemini-3.8-flash"),
        "routing_tier": state.get("last_routing_tier", "flash"),
        "pending_hitl": pending_hitl,
    }


@app.get("/api/travel/telemetry")
async def travel_telemetry_endpoint() -> dict[str, Any]:
    """Return recent structured JSON Intent/Outcome logs and OpenTelemetry spans."""
    return {
        "status": "success",
        "events": get_recent_telemetry_events(limit=30),
        "spans": get_recent_exported_spans(limit=20),
    }


@app.get("/api/travel/schemas")
async def travel_schemas_endpoint() -> dict[str, Any]:
    """Expose explicit Pydantic JSON Schemas for all travel agent tools."""
    return {
        "status": "success",
        "schemas": TOOL_JSON_SCHEMAS,
    }


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
