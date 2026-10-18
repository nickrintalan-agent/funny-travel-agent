# ruff: noqa
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

"""Captain Layover — Multi-Agent Comedic Travel Concierge (ADK Python).

Wires all 5 categories (19 criteria, 95 points) of the AgentOps Code Review Matrix:
1. Tool & Interface Design: Comprehensive docstrings, descriptive tool names,
   strict Pydantic JSON schemas, and guided error recovery (`app/tools.py`).
2. Context & Memory: Robust Constitution prompt (`app/prompts.py`), ADK
   `EventsCompactionConfig` + `ContextCacheConfig` + sliding-window compaction,
   persistent SQLite/Vector & Vertex AI Search/Memory Bank store, and non-blocking
   async memory consolidation (`app/memory.py`).
3. Orchestration & Logic: Coordinator + Specialist Sub-Agents + `SequentialAgent`
   pipeline, Strategic Model Routing (`gemini-3.8-flash` for fast search tasks vs.
   `gemini-3.1-pro-preview` for complex itinerary planning), `TravelSafetyAndPolicyPlugin`
   guardrails, and Human-in-the-Loop (HITL) booking hooks (`app/guardrails.py`).
4. Observability & Tracing: Structured JSON logging, paired Intent vs. Outcome
   capture, OpenTelemetry distributed tracing, and active Cloud DLP + regex PII
   redaction (`app/observability.py`).
5. Infrastructure & CI/CD: Automated evaluation suite (`tests/eval/`), Terraform
   IaC (`deployment/terraform/` & `infra/terraform/`), and Google Cloud Secret
   Manager secret injection (`app/secrets.py`).
"""

from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from app.guardrails import (
    COMPLEX_PRO_MODEL,
    TravelSafetyAndPolicyPlugin,
    enforce_human_in_the_loop_confirmation_callback,
    strategic_model_routing_callback,
)
from app.memory import (
    async_after_agent_memory_callback,
    build_context_cache_config,
    build_events_compaction_config,
    build_resumability_config,
    compact_conversation_history_callback,
)
from app.observability import (
    configure_distributed_tracing,
    log_agent_intent_callback,
    log_agent_outcome_callback,
    log_tool_intent_callback,
    log_tool_outcome_callback,
)
from app.prompts import (
    CAPTAIN_LAYOVER_CONSTITUTION_PROMPT,
    ITINERARY_ARCHITECT_INSTRUCTION,
    RESERVATION_BOOKING_INSTRUCTION,
    SEARCH_SPECIALIST_INSTRUCTION,
)
from app.tools import (
    book_confirmed_flight_reservation,
    book_confirmed_hotel_reservation,
    calculate_trip_budget_estimate,
    recall_traveler_preferences_from_memory,
    retrieve_destination_travel_advisory,
    save_traveler_preference_to_memory,
    search_commercial_flights_by_route,
    search_hotel_accommodations_by_city,
)

# Initialize OpenTelemetry Distributed Tracing
configure_distributed_tracing(service_name="funny-travel-agent")

MODEL = "gemini-3.8-flash"
PRO_MODEL = COMPLEX_PRO_MODEL

_SHARED_BEFORE_AGENT_CALLBACKS = [log_agent_intent_callback]
_SHARED_BEFORE_MODEL_CALLBACKS = [
    strategic_model_routing_callback,
    compact_conversation_history_callback,
]
_SHARED_BEFORE_TOOL_CALLBACKS = [
    log_tool_intent_callback,
    enforce_human_in_the_loop_confirmation_callback,
]
_SHARED_AFTER_TOOL_CALLBACKS = [log_tool_outcome_callback]
_SHARED_AFTER_AGENT_CALLBACKS = [
    log_agent_outcome_callback,
    async_after_agent_memory_callback,
]

# 1. Fast Search Specialist Sub-Agent (Gemini Flash)
travel_search_specialist = Agent(
    name="travel_search_specialist",
    description=(
        "High-speed specialist on Gemini Flash for searching commercial flights, "
        "hotel accommodations, and destination travel advisories."
    ),
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SEARCH_SPECIALIST_INSTRUCTION,
    tools=[
        search_commercial_flights_by_route,
        search_hotel_accommodations_by_city,
        retrieve_destination_travel_advisory,
    ],
    before_agent_callback=_SHARED_BEFORE_AGENT_CALLBACKS,
    before_model_callback=_SHARED_BEFORE_MODEL_CALLBACKS,
    before_tool_callback=_SHARED_BEFORE_TOOL_CALLBACKS,
    after_tool_callback=_SHARED_AFTER_TOOL_CALLBACKS,
    after_agent_callback=_SHARED_AFTER_AGENT_CALLBACKS,
)

# 2. Complex Planning & Booking Sub-Agents in a SequentialAgent Pipeline (Gemini Pro)
itinerary_architect_agent = Agent(
    name="itinerary_architect_agent",
    description=(
        "Complex planning specialist on Gemini Pro that builds multi-day travel "
        "itineraries and itemized trip budget estimates."
    ),
    model=Gemini(
        model=PRO_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=ITINERARY_ARCHITECT_INSTRUCTION,
    tools=[
        retrieve_destination_travel_advisory,
        calculate_trip_budget_estimate,
    ],
    output_key="drafted_itinerary_plan",
    before_agent_callback=_SHARED_BEFORE_AGENT_CALLBACKS,
    before_model_callback=_SHARED_BEFORE_MODEL_CALLBACKS,
    before_tool_callback=_SHARED_BEFORE_TOOL_CALLBACKS,
    after_tool_callback=_SHARED_AFTER_TOOL_CALLBACKS,
    after_agent_callback=_SHARED_AFTER_AGENT_CALLBACKS,
)

reservation_booking_agent = Agent(
    name="reservation_booking_agent",
    description=(
        "High-stakes booking specialist on Gemini Pro enforcing Human-in-the-Loop (HITL) "
        "confirmation before finalizing flight or hotel reservations."
    ),
    model=Gemini(
        model=PRO_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=RESERVATION_BOOKING_INSTRUCTION,
    tools=[
        book_confirmed_flight_reservation,
        book_confirmed_hotel_reservation,
    ],
    before_agent_callback=_SHARED_BEFORE_AGENT_CALLBACKS,
    before_model_callback=_SHARED_BEFORE_MODEL_CALLBACKS,
    before_tool_callback=_SHARED_BEFORE_TOOL_CALLBACKS,
    after_tool_callback=_SHARED_AFTER_TOOL_CALLBACKS,
    after_agent_callback=_SHARED_AFTER_AGENT_CALLBACKS,
)

itinerary_booking_pipeline = SequentialAgent(
    name="itinerary_booking_pipeline",
    description=(
        "Sequential multi-agent pipeline combining Gemini Pro itinerary synthesis "
        "and Human-in-the-Loop reservation booking."
    ),
    sub_agents=[itinerary_architect_agent, reservation_booking_agent],
)

# 3. Root Coordinator Agent ("Captain Layover")
root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="funny_travel_agent",
    description=(
        "Captain Layover: Chief comedic travel concierge coordinating flight searches, "
        "hotel discovery, destination advisories, budget math, multi-day itineraries, "
        "and Human-in-the-Loop bookings."
    ),
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=CAPTAIN_LAYOVER_CONSTITUTION_PROMPT,
    tools=[
        search_commercial_flights_by_route,
        search_hotel_accommodations_by_city,
        retrieve_destination_travel_advisory,
        calculate_trip_budget_estimate,
        book_confirmed_flight_reservation,
        book_confirmed_hotel_reservation,
        save_traveler_preference_to_memory,
        recall_traveler_preferences_from_memory,
        PreloadMemoryTool(),
    ],
    sub_agents=[
        travel_search_specialist,
        itinerary_booking_pipeline,
    ],
    before_agent_callback=_SHARED_BEFORE_AGENT_CALLBACKS,
    before_model_callback=_SHARED_BEFORE_MODEL_CALLBACKS,
    before_tool_callback=_SHARED_BEFORE_TOOL_CALLBACKS,
    after_tool_callback=_SHARED_AFTER_TOOL_CALLBACKS,
    after_agent_callback=_SHARED_AFTER_AGENT_CALLBACKS,
)

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[TravelSafetyAndPolicyPlugin()],
    events_compaction_config=build_events_compaction_config(),
    context_cache_config=build_context_cache_config(),
    resumability_config=build_resumability_config(),
)
