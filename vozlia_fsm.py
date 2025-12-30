"""
vozlia_fsm.py

Finite State Machine for Vozlia using the `transitions` library.

This module is intentionally narrow in scope:

- It classifies each caller utterance into a simple intent
  (check email, greeting, small talk, generic help, etc.)
- It manages a small set of states (idle, email_intent, small_talk, generic_help).
- For "skill" intents (like checking email), it emits a `backend_call`
  payload that the backend can execute (e.g., Gmail summary).
- It returns a `spoken_reply` that can be read to the caller.

The heavy lifting (Gmail API, weather, tasks, etc.) is done in your
FastAPI backend, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transitions import Machine


# -------------------- Intent labels --------------------


INTENT_GREETING = "greeting"
INTENT_CHECK_EMAIL = "check_email"
INTENT_SMALL_TALK = "small_talk"
INTENT_GENERIC_HELP = "generic_help"
INTENT_MEMORY_STORE = "memory_store"
INTENT_MEMORY_RECALL = "memory_recall"
INTENT_UNKNOWN = "unknown"


# -------------------- FSM states --------------------


FSM_STATES = [
    "idle",
    "email_intent",
    "small_talk",
    "generic_help",
]


@dataclass
class VozliaFSM:
    """
    A very lightweight FSM wrapper around `transitions.Machine`.

    Public API:
        fsm = VozliaFSM()
        result = fsm.handle_utterance("Check my unread emails", context={...})

    `result` is a dict:
      {
        "intent": "check_email",
        "previous_state": "idle",
        "next_state": "email_intent",
        "spoken_reply": "Sure, I'll take a quick look at your recent unread emails.",
        "backend_call": {
            "type": "gmail_summary",
            "params": {
                "query": "is:unread",
                "max_results": 20
            }
        },
        "context": {...}
      }
    """
    greeting_text: str | None = None
    # transitions Machine will attach itself and manage `state` attr
    machine: Machine = field(init=False, repr=False)
    state: str = field(init=False, default="idle")

    def __post_init__(self) -> None:
        # Initialize the underlying transitions.Machine
        self.machine = Machine(
            model=self,
            states=FSM_STATES,
            initial="idle",
            ignore_invalid_triggers=True,  # don't explode on unknown triggers
        )

        # Generic transitions to move into specific "intent" states
        self.machine.add_transition("to_email_intent", "*", "email_intent")
        self.machine.add_transition("to_small_talk", "*", "small_talk")
        self.machine.add_transition("to_generic_help", "*", "generic_help")
        self.machine.add_transition("reset_to_idle", "*", "idle")

    # -------------------- Public API --------------------

    def handle_utterance(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint.

        - Classifies intent from `text`
        - Transitions FSM state accordingly
        - Produces:
            - spoken_reply (string)
            - intent (string)
            - previous_state / next_state
            - backend_call (dict or None)
            - context (echoed)

        This function does NOT talk to any external APIs – it just
        decides *what should happen* and lets the backend do the work.
        """
        context = context or {}
        original_state = self.state

        cleaned = (text or "").strip()
        lowered = cleaned.lower()

        if not lowered:
            # Blank / noise – keep it simple
            intent = INTENT_UNKNOWN
            spoken_reply = "I didn’t quite catch that. Could you please repeat?"
            backend_call = None
            # No need to change state
            return {
                "intent": intent,
                "previous_state": original_state,
                "next_state": self.state,
                "spoken_reply": spoken_reply,
                "backend_call": backend_call,
                "context": context,
            }

        # 1) Classify the intent
        intent = self._classify_intent(lowered)

        # 2) Drive state transitions + decide backend_call + spoken_reply
        backend_call: Optional[Dict[str, Any]] = None
        spoken_reply: str

        if intent == INTENT_CHECK_EMAIL:
            self.to_email_intent()
            backend_call, spoken_reply = self._handle_email_intent(lowered)

        elif intent == INTENT_GREETING:
            self.reset_to_idle()
            spoken_reply = self._handle_greeting(cleaned)

        elif intent == INTENT_SMALL_TALK:
            self.to_small_talk()
            spoken_reply = self._handle_small_talk(cleaned)


        elif intent == INTENT_MEMORY_STORE:
            # Let backend persist the fact; keep spoken reply short as a fallback.
            self.to_small_talk()
            backend_call = {"type": "memory_store", "text": cleaned}
            spoken_reply = "Got it — I’ll remember that."

        elif intent == INTENT_MEMORY_RECALL:
            self.to_small_talk()
            backend_call = {"type": "memory_recall"}
            spoken_reply = "Let me check what you told me."
        elif intent == INTENT_GENERIC_HELP:
            self.to_generic_help()
            spoken_reply = (
                "Sure, tell me what you’d like help with and I’ll take care of it."
            )

        else:  # INTENT_UNKNOWN
            # For unknowns, keep state simple: go back to idle
            self.reset_to_idle()
            spoken_reply = (
                "I’m not entirely sure what you meant. "
                "Could you rephrase that or give me a bit more detail?"
            )

        # 3) Build result payload
        result = {
            "intent": intent,
            "previous_state": original_state,
            "next_state": self.state,
            "spoken_reply": spoken_reply,
            "backend_call": backend_call,
            "context": context,
        }
        return result

    # -------------------- Intent classification --------------------

    def _classify_intent(self, lowered: str) -> str:
        """
        Extremely simple rule-based classifier.

        This is intentionally conservative: the model (GPT/Realtime)
        will already be very smart. The FSM just nudges clear patterns
        into explicit backend actions.
        """

        # Memory store / recall (long-term memory)
        # Store: "remember my favorite color is green"
        if any(p in lowered for p in [
            "please remember",
            "remember that",
            "remember this",
            "note that",
            "my favorite color is",
            "my favourite colour is",
            "my favorite colour is",
            "my favourite color is",
        ]):
            return INTENT_MEMORY_STORE

        # Recall: "what did I say my favorite color was 5 minutes ago?"
        if any(p in lowered for p in [
            "what did i say",
            "remind me",
            "do you remember",
            "what was my",
            "what is my",
        ]) and ("favorite color" in lowered or "favourite colour" in lowered or "favorite colour" in lowered):
            return INTENT_MEMORY_RECALL
        if ("favorite color" in lowered or "favourite colour" in lowered or "favorite colour" in lowered) and ("ago" in lowered or "last time" in lowered):
            return INTENT_MEMORY_RECALL

        # Check email / inbox / unread
        email_keywords = [
            "email",
            "emails",
            "inbox",
            "gmail",
            "unread",
            "new mail",
            "new emails",
            "check my mail",
            "check my inbox",
            "latest emails",
        ]
        if any(k in lowered for k in email_keywords):
            return INTENT_CHECK_EMAIL

        # Greetings
        greeting_keywords = [
            "hello",
            "hi ",
            "hi,",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(lowered.startswith(k) for k in greeting_keywords):
            return INTENT_GREETING

        # Small talk topics
        small_talk_keywords = [
            "how are you",
            "how's it going",
            "how are you doing",
            "who are you",
            "what are you",
            "tell me about yourself",
        ]
        if any(k in lowered for k in small_talk_keywords):
            return INTENT_SMALL_TALK

        # Generic help phrases
        generic_help_keywords = [
            "i need help",
            "can you help",
            "i have a question",
            "i need some assistance",
            "help me with",
        ]
        if any(k in lowered for k in generic_help_keywords):
            return INTENT_GENERIC_HELP

        return INTENT_UNKNOWN

    # -------------------- Handlers for each intent --------------------

    def _handle_email_intent(
        self,
        lowered: str,
    ) -> tuple[Dict[str, Any], str]:
        """
        Decide *how* to query Gmail based on the utterance.

        Returns:
            backend_call, spoken_reply
        """
        # Default Gmail query and max_results
        gmail_query = None
        max_results = 20

        # Common patterns: unread, today, this week, etc.
        if "unread" in lowered or "new" in lowered or "newest" in lowered:
            gmail_query = "is:unread"

        # Time windows
        if "today" in lowered:
            # e.g. "emails from today" -> use newer_than:1d
            if gmail_query:
                gmail_query = f"{gmail_query} newer_than:1d"
            else:
                gmail_query = "newer_than:1d"

        if "yesterday" in lowered:
            # GMail doesn't have a native "yesterday" filter,
            # but we can approximate with 2 days if needed.
            if gmail_query:
                gmail_query = f"{gmail_query} newer_than:2d"
            else:
                gmail_query = "newer_than:2d"

        if "this week" in lowered or "past week" in lowered:
            if gmail_query:
                gmail_query = f"{gmail_query} newer_than:7d"
            else:
                gmail_query = "newer_than:7d"

        # If caller mentions "only a few" or "top couple", reduce max_results
        if "few" in lowered or "couple" in lowered or "top 3" in lowered:
            max_results = 5
        elif "top 10" in lowered:
            max_results = 10

        backend_call = {
            "type": "gmail_summary",
            "params": {
                "query": gmail_query,
                "max_results": max_results,
                # NOTE: account_id is resolved in the backend
                # based on payload.account_id or default Gmail account.
            },
        }

        if gmail_query == "is:unread":
            spoken_reply = (
                "Sure, I’ll take a quick look at your unread emails and summarize them."
            )
        elif gmail_query:
            spoken_reply = (
                "Okay, I’ll check your recent emails that match what you asked for."
            )
        else:
            spoken_reply = (
                "Got it, I’ll take a quick look at your recent emails and summarize them."
            )

        return backend_call, spoken_reply

    def _handle_greeting(self, cleaned: str) -> str:
        # If the app sets a portal-controlled greeting, prefer it.
        if getattr(self, "greeting_text", None):
            gt = str(self.greeting_text).strip()
            if gt:
                return gt
        return (
            "Hi, this is Vozlia. I’m your AI assistant. "
            "What would you like help with today?"
        )

    def _handle_small_talk(self, cleaned: str) -> str:
        # You can make this more playful later if you’d like
        return (
            "I’m doing well and ready to help. "
            "Tell me what you’d like me to do for you."
        )