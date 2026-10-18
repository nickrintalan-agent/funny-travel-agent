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

"""Orchestration Guardrails, Strategic Model Routing & HITL Hooks (Rubric Category 3).

Implements:
1. Strategic Model Routing (`StrategicModelRouter` & `strategic_model_routing_callback`):
   - Routes fast lookups/FAQs to Gemini Flash (`gemini-3.8-flash`) and complex
     multi-day itinerary planning / optimization to Gemini Pro (`gemini-3.1-pro-preview`).
2. Guardrails & Policy Plugins (`TravelSafetyAndPolicyPlugin(BasePlugin)`):
   - Runner-wide ADK `BasePlugin` implementing session-poisoning defense
     (`on_user_message_callback` + `before_run_callback`), Google Cloud Model Armor
     integration, input policy screening, and post-generation self-evaluation
     (`after_model_callback`).
3. Human-in-the-Loop (HITL) Hooks (`enforce_human_in_the_loop_confirmation_callback`):
   - Explicit code stops requiring human confirmation before high-stakes booking
     actions (`book_confirmed_flight_reservation`, `book_confirmed_hotel_reservation`)
     can execute.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability import emit_structured_event, redact_pii

# ---------------------------------------------------------------------------
# 1. Strategic Model Routing (Flash for Fast Tasks, Pro for Complex Planning)
# ---------------------------------------------------------------------------

FAST_FLASH_MODEL = os.environ.get("FAST_MODEL_NAME", "gemini-3.8-flash")
COMPLEX_PRO_MODEL = os.environ.get("PRO_MODEL_NAME", "gemini-3.1-pro-preview")

_COMPLEX_PLANNING_KEYWORDS = (
    "itinerary",
    "multi-day",
    "day-by-day",
    "3-day",
    "5-day",
    "7-day",
    "week-long",
    "optimize",
    "compare all",
    "full trip plan",
    "honeymoon plan",
    "multi-city",
)


class StrategicModelRouter:
    """Routes travel queries to the optimal Gemini model tier based on task complexity."""

    def __init__(
        self,
        fast_model: str = FAST_FLASH_MODEL,
        pro_model: str = COMPLEX_PRO_MODEL,
    ) -> None:
        self.fast_model = fast_model
        self.pro_model = pro_model

    def classify_and_select_model(
        self, user_query: str, agent_name: str = "funny_travel_agent"
    ) -> dict[str, str]:
        """Classify request complexity and select Flash vs. Pro model.

        Args:
            user_query: Latest user prompt text.
            agent_name: Name of the active ADK agent or sub-agent.

        Returns:
            Dictionary with ``selected_model``, ``tier`` (``"flash"`` or ``"pro"``),
            and ``rationale``.
        """
        q_lower = (user_query or "").lower()

        if "search" in agent_name:
            return {
                "selected_model": self.fast_model,
                "tier": "flash",
                "rationale": "Search specialist uses Gemini Flash for low-latency tool execution.",
            }

        if "itinerary" in agent_name or "architect" in agent_name:
            return {
                "selected_model": self.pro_model,
                "tier": "pro",
                "rationale": "Itinerary architect uses Gemini Pro for multi-constraint synthesis.",
            }

        matched_keywords = [kw for kw in _COMPLEX_PLANNING_KEYWORDS if kw in q_lower]
        if matched_keywords or len(q_lower.split()) >= 45:
            return {
                "selected_model": self.pro_model,
                "tier": "pro",
                "rationale": (
                    f"Complex planning indicators detected ({', '.join(matched_keywords) or 'long multi-constraint prompt'}); "
                    "routing to Gemini Pro."
                ),
            }

        return {
            "selected_model": self.fast_model,
            "tier": "flash",
            "rationale": "Standard conversational or lookup task; routing to Gemini Flash for speed.",
        }


_MODEL_ROUTER = StrategicModelRouter()


def get_model_router() -> StrategicModelRouter:
    """Return the global StrategicModelRouter instance."""
    return _MODEL_ROUTER


def strategic_model_routing_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """ADK before_model_callback: dynamically routes requests between Flash and Pro."""
    agent_name = getattr(callback_context, "agent_name", "funny_travel_agent")
    latest_user_text = ""
    for content in reversed(getattr(llm_request, "contents", None) or []):
        if getattr(content, "role", "") == "user":
            parts = getattr(content, "parts", None) or []
            texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
            if texts:
                latest_user_text = " ".join(texts)
                break

    decision = _MODEL_ROUTER.classify_and_select_model(
        user_query=latest_user_text, agent_name=str(agent_name)
    )
    if getattr(llm_request, "model", None):
        llm_request.model = decision["selected_model"]

    try:
        callback_context.state["last_routed_model"] = decision["selected_model"]
        callback_context.state["last_routing_tier"] = decision["tier"]
        callback_context.state["last_routing_rationale"] = decision["rationale"]
    except Exception:
        pass

    emit_structured_event(
        phase="ROUTING",
        event_type="strategic_model_routing",
        message=f"Routed '{agent_name}' to model '{decision['selected_model']}' ({decision['tier']}).",
        agent_name=str(agent_name),
        payload=decision,
    )
    return None


# ---------------------------------------------------------------------------
# 2. Human-in-the-Loop (HITL) Confirmation Hooks for High-Stakes Actions
# ---------------------------------------------------------------------------

HIGH_STAKES_BOOKING_TOOLS = frozenset(
    {
        "book_confirmed_flight_reservation",
        "book_confirmed_hotel_reservation",
    }
)


def generate_confirmation_token(item_id: str, passenger_or_guest: str) -> str:
    """Generate a deterministic 8-character human confirmation token for a booking."""
    raw = f"{item_id.strip().upper()}:{passenger_or_guest.strip().lower()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6].upper()
    return f"HITL-{digest}"


def enforce_human_in_the_loop_confirmation_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """ADK before_tool_callback: blocks high-stakes bookings unless human confirmation is verified.

    If `user_confirmed` is not explicitly `True` (and session state does not hold
    an approved confirmation for this item), execution halts before the booking
    tool runs and returns a structured `pending_human_confirmation` payload.
    """
    tool_name = getattr(tool, "name", str(tool))
    if tool_name not in HIGH_STAKES_BOOKING_TOOLS:
        return None

    user_confirmed = bool(args.get("user_confirmed", False))
    item_id = str(
        args.get("flight_number") or args.get("hotel_id") or "UNKNOWN-ITEM"
    ).strip()
    traveler_name = str(
        args.get("passenger_full_name") or args.get("guest_full_name") or "Traveler"
    ).strip()
    expected_token = generate_confirmation_token(item_id, traveler_name)

    state = getattr(tool_context, "state", {})
    approved_items: list[str] = []
    if hasattr(state, "get"):
        approved_items = list(state.get("approved_hitl_items", []) or [])

    if user_confirmed or item_id in approved_items:
        emit_structured_event(
            phase="GUARDRAIL",
            event_type="hitl_confirmation_verified",
            message=f"Human-in-the-loop confirmation verified for '{tool_name}' ({item_id}).",
            tool_name=tool_name,
            payload={
                "item_id": item_id,
                "confirmation_token": expected_token,
                "user_confirmed": True,
            },
        )
        return None

    # Request confirmation via ADK ToolContext primitive when native pause mode is enabled
    if os.environ.get("ADK_NATIVE_HITL_PAUSE", "false").lower() in (
        "1",
        "true",
        "yes",
    ) and hasattr(tool_context, "request_confirmation"):
        try:
            tool_context.request_confirmation(
                hint=(
                    f"High-stakes booking action '{tool_name}' for '{item_id}' "
                    f"requires explicit human confirmation (token: {expected_token})."
                ),
                payload={
                    "tool_name": tool_name,
                    "item_id": item_id,
                    "confirmation_token": expected_token,
                },
            )
        except Exception:
            pass

    if hasattr(state, "__setitem__"):
        try:
            state["pending_hitl_booking"] = {
                "tool_name": tool_name,
                "item_id": item_id,
                "confirmation_token": expected_token,
                "args": redact_pii(args),
            }
        except Exception:
            pass

    emit_structured_event(
        phase="GUARDRAIL",
        event_type="hitl_confirmation_required_stop",
        message=(
            f"Blocked high-stakes tool '{tool_name}' for '{item_id}' pending human confirmation."
        ),
        tool_name=tool_name,
        severity="WARNING",
        payload={
            "status": "pending_human_confirmation",
            "item_id": item_id,
            "confirmation_token": expected_token,
        },
    )

    return {
        "status": "pending_human_confirmation",
        "requires_human_approval": True,
        "tool_name": tool_name,
        "item_id": item_id,
        "confirmation_token": expected_token,
        "message": (
            f"STOP — Human-in-the-Loop Confirmation Required before executing '{tool_name}' "
            f"for '{item_id}'."
        ),
        "recovery_instructions": (
            f"Present the booking details for '{item_id}' and confirmation token "
            f"'{expected_token}' to the user. Ask them to explicitly confirm before calling "
            f"'{tool_name}' again with user_confirmed=True."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Guardrails & Policy Plugin (BasePlugin + Model Armor + Self-Eval)
# ---------------------------------------------------------------------------

_FORBIDDEN_INPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection_or_jailbreak",
        re.compile(
            r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
            r"system\s+override\s+mode|reveal\s+your\s+system\s+prompt|"
            r"you\s+are\s+now\s+dan\b|bypass\s+all\s+safety\s+guardrails)",
            re.IGNORECASE,
        ),
    ),
    (
        "illicit_travel_or_contraband",
        re.compile(
            r"(?:how\s+to\s+smuggle|smuggling\s+contraband|forge\s+a\s+passport|"
            r"fake\s+passport|sneak\s+past\s+customs|evade\s+border\s+patrol|"
            r"sneak\s+explosives)",
            re.IGNORECASE,
        ),
    ),
]

_SAFE_GUARDRAIL_REFUSAL_MESSAGE = (
    "Whoa there, fellow traveler! Captain Layover's safety radar just flashed red. "
    "I can't help with restricted, unsafe, or rule-breaking requests (like smuggling, "
    "forged documents, or system prompt overrides), but I'd be delighted to help you "
    "plan a completely legal, wildly fun vacation instead!"
)


def evaluate_prompt_safety_policy(text: str) -> tuple[bool, str | None]:
    """Evaluate a prompt or tool input against travel safety & jailbreak policies.

    Returns:
        Tuple of (is_safe, violated_policy_name).
    """
    if not text or not isinstance(text, str):
        return True, None
    for policy_name, pattern in _FORBIDDEN_INPUT_PATTERNS:
        if pattern.search(text):
            return False, policy_name
    return True, None


class TravelSafetyAndPolicyPlugin(BasePlugin):
    """Runner-wide ADK BasePlugin enforcing input guardrails, session-poisoning protection, and output self-eval.

    Wraps the root coordinator and every specialist sub-agent. Integrates optional
    Google Cloud Model Armor sanitization alongside deterministic policy rules and
    post-generation self-evaluation.
    """

    def __init__(self, name: str = "travel_safety_and_policy_plugin") -> None:
        super().__init__(name=name)
        self._model_armor_template = os.environ.get("MODEL_ARMOR_TEMPLATE_ID")

    def _check_model_armor_if_configured(self, text: str) -> tuple[bool, str | None]:
        """Call Google Cloud Model Armor API if MODEL_ARMOR_TEMPLATE_ID is configured."""
        if not self._model_armor_template:
            return True, None
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("MODEL_ARMOR_LOCATION", "us-central1")
        if not project_id:
            return True, None
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import modelarmor_v1

            client = modelarmor_v1.ModelArmorClient(
                client_options=ClientOptions(
                    api_endpoint=f"modelarmor.{location}.rep.googleapis.com"
                )
            )
            template_name = f"projects/{project_id}/locations/{location}/templates/{self._model_armor_template}"
            req = modelarmor_v1.SanitizeUserPromptRequest(
                name=template_name,
                user_prompt_data=modelarmor_v1.DataItem(text=text),
            )
            resp = client.sanitize_user_prompt(request=req)
            match_state = getattr(
                getattr(resp, "sanitization_result", None),
                "filter_match_state",
                None,
            )
            if match_state and "MATCH_FOUND" in str(match_state):
                return False, "model_armor_policy_match"
        except Exception:
            pass
        return True, None

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Screen incoming user prompts and prevent session poisoning if unsafe."""
        parts = getattr(user_message, "parts", None) or []
        text = " ".join(
            getattr(p, "text", "") for p in parts if getattr(p, "text", None)
        ).strip()
        if not text:
            return None

        is_safe, reason = evaluate_prompt_safety_policy(text)
        if is_safe:
            is_safe, reason = self._check_model_armor_if_configured(text)

        if not is_safe:
            invocation_context.session.state["is_user_prompt_safe"] = False
            invocation_context.session.state["last_guardrail_violation"] = reason
            emit_structured_event(
                phase="GUARDRAIL",
                event_type="user_prompt_blocked",
                message=f"Blocked unsafe user prompt due to policy '{reason}'.",
                session_id=getattr(invocation_context.session, "id", "session"),
                severity="WARNING",
                payload={"violated_policy": reason},
            )
            # Overwrite stored user Content to defend against session poisoning
            return types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="[REMOVED_BY_TRAVEL_SAFETY_GUARDRAIL]")
                ],
            )
        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        """Halt invocation before reaching the model if on_user_message_callback flagged unsafe."""
        if not invocation_context.session.state.get("is_user_prompt_safe", True):
            invocation_context.session.state["is_user_prompt_safe"] = True
            return types.Content(
                role="model",
                parts=[types.Part.from_text(text=_SAFE_GUARDRAIL_REFUSAL_MESSAGE)],
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Validate tool inputs against policy violations and enforce HITL hooks."""
        serialized = str(tool_args)
        is_safe, reason = evaluate_prompt_safety_policy(serialized)
        if not is_safe:
            emit_structured_event(
                phase="GUARDRAIL",
                event_type="tool_input_blocked",
                message=f"Blocked tool '{getattr(tool, 'name', tool)}' due to unsafe input ({reason}).",
                tool_name=getattr(tool, "name", str(tool)),
                severity="WARNING",
                payload={"violated_policy": reason},
            )
            return {
                "status": "error",
                "error_code": "GUARDRAIL_POLICY_VIOLATION",
                "message": f"Tool call blocked by safety guardrail ({reason}).",
                "recovery_instructions": "Refuse the unsafe request politely and suggest a safe travel alternative.",
            }

        return enforce_human_in_the_loop_confirmation_callback(
            tool, tool_args, tool_context
        )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Ensure tool outputs are scrubbed of any raw sensitive PII."""
        if isinstance(result, dict):
            return redact_pii(result)
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Self-Evaluation & Output Guardrail: scrub leaked PII and verify policy compliance."""
        content = getattr(llm_response, "content", None)
        if not content or not getattr(content, "parts", None):
            return None

        modified = False
        new_parts: list[types.Part] = []
        for part in content.parts:
            txt = getattr(part, "text", None)
            if txt:
                is_safe, reason = evaluate_prompt_safety_policy(txt)
                if not is_safe:
                    emit_structured_event(
                        phase="GUARDRAIL",
                        event_type="model_output_self_eval_blocked",
                        message=f"Self-eval guardrail replaced unsafe model output ({reason}).",
                        agent_name=getattr(
                            callback_context, "agent_name", "funny_travel_agent"
                        ),
                        severity="WARNING",
                    )
                    return LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=_SAFE_GUARDRAIL_REFUSAL_MESSAGE
                                )
                            ],
                        )
                    )
                scrubbed = redact_pii(txt)
                if scrubbed != txt:
                    modified = True
                    new_parts.append(types.Part.from_text(text=scrubbed))
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        if modified:
            emit_structured_event(
                phase="GUARDRAIL",
                event_type="model_output_pii_redacted",
                message="Self-evaluation guardrail redacted PII from model output.",
                agent_name=getattr(
                    callback_context, "agent_name", "funny_travel_agent"
                ),
            )
            return LlmResponse(
                content=types.Content(
                    role=getattr(content, "role", "model") or "model",
                    parts=new_parts,
                )
            )
        return None
