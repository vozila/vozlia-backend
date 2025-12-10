# vozlia_fsm.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from transitions import Machine


class TaskType(Enum):
    NONE = auto()
    EMAIL_UNREAD_SUMMARY = auto()
    # Later: CALENDAR_SUMMARY, CREATE_MEETING, WEATHER_LOOKUP, etc.


@dataclass
class CallContext:
    """
    Per-call server-side context.
    This is your 'memory' for that phone call.
    """
    call_id: str
    active_task: TaskType = TaskType.NONE
    slots: Dict[str, Any] = field(default_factory=dict)
    last_user_utterance: str = ""
    last_assistant_utterance: str = ""


@dataclass
class FSMAction:
    """
    What the FSM wants the backend to do right now.
    """
    speak: Optional[str] = None          # text for Vozlia to say
    api_call: Optional[Dict[str, Any]] = None  # e.g. {"type": "gmail_unread_summary", "params": {...}}


class CallFSM:
    """
    A finite state machine for a single call, powered by the 'transitions' library.
    """

    # Define state names (simple strings for transitions)
    states = [
        "idle",                 # no active task
        "email_unread_fetching",
        "email_unread_done",
    ]

    def __init__(self, call_id: str):
        self.ctx = CallContext(call_id=call_id)
        self._pending_action: Optional[FSMAction] = None

        # Machine from 'transitions'
        self.machine = Machine(
            model=self,
            states=CallFSM.states,
            initial="idle",
            ignore_invalid_triggers=True,  # don't crash on bad trigger
        )

        # Transitions for email-unread flow
        self.machine.add_transition(
            trigger="start_email_unread_flow",
            source="idle",
            dest="email_unread_fetching",
            after="_after_start_email_unread_flow",
        )

        self.machine.add_transition(
            trigger="finish_email_unread_flow",
            source="email_unread_fetching",
            dest="email_unread_done",
            after="_after_finish_email_unread_flow",
        )

        self.machine.add_transition(
            trigger="reset",
            source="*",
            dest="idle",
            after="_after_reset",
        )

    # -------------------------------------------------------------------------
    # Helpers to get / clear the last action (FSMAction)
    # -------------------------------------------------------------------------

    def pop_action(self) -> Optional[FSMAction]:
        action = self._pending_action
        self._pending_action = None
        return action

    def _set_action(self, speak: Optional[str] = None, api_call: Optional[Dict[str, Any]] = None):
        self._pending_action = FSMAction(speak=speak, api_call=api_call)

    # -------------------------------------------------------------------------
    # Callbacks wired into transitions
    # -------------------------------------------------------------------------

    def _after_start_email_unread_flow(self):
        """
        We just transitioned: idle -> email_unread_fetching
        Decide what to say and which API call to request.
        """
        self.ctx.active_task = TaskType.EMAIL_UNREAD_SUMMARY

        speak = (
            "Okay, I’ll take a quick look at your unread email and summarize "
            "what looks important."
        )

        # Let the backend know exactly what to do.
        api_call = {
            "type": "gmail_unread_summary",
            "params": {
                "query": "is:unread",
                "max_results": 20,
            },
        }

        self._set_action(speak=speak, api_call=api_call)

    def _after_finish_email_unread_flow(self):
        """
        This is called after we move email_unread_fetching -> email_unread_done.
        The actual summary text will be provided via handle_api_result().
        """
        # Nothing to do here; handle_api_result will set the 'speak' message.
        pass

    def _after_reset(self):
        """
        Reset context after finishing a task.
        """
        self.ctx.active_task = TaskType.NONE
        self.ctx.slots.clear()
        self._set_action(
            speak="All set. What else would you like help with?",
            api_call=None,
        )

    # -------------------------------------------------------------------------
    # Public interface: called by your backend
    # -------------------------------------------------------------------------

    def handle_user_utterance(self, text: str) -> Optional[FSMAction]:
        """
        Backend calls this when you have a clean user utterance (transcript).
        This method may:
          - Trigger a state transition
          - Set a 'speak' message
          - Request an API call
          - Or do nothing (return None) and let the LLM handle free-form chat.
        """
        self.ctx.last_user_utterance = text
        lower = text.lower().strip()

        # When idle, try to detect structured intents.
        if self.state == "idle":
            if "email" in lower and ("unread" in lower or "new" in lower):
                # Kick off unread-email flow.
                self.start_email_unread_flow()
                return self.pop_action()

            # No structured task recognized → let the LLM handle it.
            return None

        # If we've already done the unread email summary,
        # handle simple follow-ups in a structured way.
        if self.state == "email_unread_done":
            if "anything important" in lower:
                self._set_action(
                    speak=(
                        "From what I saw, the key items were billing and account notices, "
                        "donation requests, and sales promotions. I can read one in more "
                        "detail if you’d like."
                    )
                )
                return self.pop_action()

            if lower in {"thanks", "that’s all", "that's all", "no thanks"}:
                self.reset()
                return self.pop_action()

            # Otherwise let LLM handle, but keep state.
            return None

        # Default: no special behavior
        return None

    def handle_api_result(self, api_type: str, result: Dict[str, Any]) -> Optional[FSMAction]:
        """
        Backend calls this after it completes an API call that the FSM requested.
        For example: Gmail unread summary.
        """
        if api_type == "gmail_unread_summary":
            summary = result.get("summary") or (
                "I wasn’t able to get a detailed summary of your unread email."
            )

            # Move to 'done' state
            self.finish_email_unread_flow()

            # Speak the summary
            self._set_action(speak=summary)
            return self.pop_action()

        # Unknown API type for now.
        return None
