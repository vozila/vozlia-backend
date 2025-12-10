# vozlia_fsm.py
"""
Vozlia Finite State Machine (FSM)
--------------------------------
This FSM does lightweight intent classification and produces
structured results telling the backend what to do.

It does NOT call any external APIs — it only decides WHAT should happen.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ----------------------------
# ENUMS FOR STATES & INTENTS
# ----------------------------

class State(Enum):
    IDLE = auto()
    EMAIL_SUMMARY = auto()
    GENERAL_QUERY = auto()
    GREETING = auto()
    SMALL_TALK = auto()
    UNKNOWN = auto()


class Intent(Enum):
    GREETING = auto()
    CHECK_EMAIL_UNREAD = auto()
    CHECK_EMAIL_ALL = auto()
    GENERIC_EMAIL_QUERY = auto()
    GENERAL_KNOWLEDGE_QUESTION = auto()
    WEATHER = auto()
    LOCATION_LOOKUP = auto()
    SMALL_TALK = auto()
    UNKNOWN = auto()


# ----------------------------
# RESULT OBJECT
# ----------------------------

@dataclass
class FSMResult:
    intent: Intent
    next_state: State
    spoken_reply: str
    backend_call: Optional[Dict[str, Any]] = field(default_factory=dict)
    raw_debug: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# MAIN FSM LOGIC
# ----------------------------

class VozliaFSM:
    def __init__(self):
        self.state = State.IDLE

    # ------------------------
    # Intent Classifier
    # ------------------------
    def classify_intent(self, text: str) -> Intent:
        t = text.lower()

        # ---- Greetings ----
        if any(x in t for x in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            return Intent.GREETING

        # ---- Email intents ----
        if "email" in t or "gmail" in t:
            if "unread" in t:
                return Intent.CHECK_EMAIL_UNREAD
            if any(x in t for x in ["check", "show", "list", "read"]):
                return Intent.CHECK_EMAIL_ALL
            return Intent.GENERIC_EMAIL_QUERY

        # ---- Weather intent ----
        if any(x in t for x in ["weather", "temperature", "forecast"]):
            return Intent.WEATHER

        # ---- Location lookup ----
        if any(x in t for x in ["nearest", "closest", "nearby"]):
            return Intent.LOCATION_LOOKUP

        # ---- General knowledge ----
        if any(x in t for x in ["who", "what", "when", "where", "why", "how"]):
            return Intent.GENERAL_KNOWLEDGE_QUESTION

        # ---- Small talk ----
        if any(x in t for x in ["how are you", "what’s up", "whats up"]):
            return Intent.SMALL_TALK

        return Intent.UNKNOWN

    # ------------------------
    # Main Handler
    # ------------------------
    def handle_utterance(self, text: str) -> Dict[str, Any]:
        intent = self.classify_intent(text)

        # Attach debug info
        debug = {
            "input": text,
            "intent_detected": intent.name,
            "previous_state": self.state.name,
        }

        # Intent routing
        if intent == Intent.GREETING:
            self.state = State.GREETING
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Hi! How can I help you today?",
                backend_call=None,
                raw_debug=debug
            ).__dict__

        # ---- Email: unread summary ----
        if intent == Intent.CHECK_EMAIL_UNREAD:
            self.state = State.EMAIL_SUMMARY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Let me check your unread emails.",
                backend_call={
                    "type": "gmail_summary",
                    "params": {
                        "query": "is:unread",
                        "max_results": 20
                    }
                },
                raw_debug=debug
            ).__dict__

        # ---- Email: general inbox ----
        if intent == Intent.CHECK_EMAIL_ALL:
            self.state = State.EMAIL_SUMMARY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Checking your inbox now.",
                backend_call={
                    "type": "gmail_summary",
                    "params": {
                        "query": None,
                        "max_results": 20
                    }
                },
                raw_debug=debug
            ).__dict__

        # ---- Generic email inquiries ----
        if intent == Intent.GENERIC_EMAIL_QUERY:
            self.state = State.EMAIL_SUMMARY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Let me review your recent email activity.",
                backend_call={
                    "type": "gmail_summary",
                    "params": {
                        "query": None,
                        "max_results": 20
                    }
                },
                raw_debug=debug
            ).__dict__

        # ---- Weather ----
        if intent == Intent.WEATHER:
            self.state = State.GENERAL_QUERY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Sure, let me look up the weather for you.",
                backend_call={
                    "type": "weather_lookup",
                    "params": {}
                },
                raw_debug=debug
            ).__dict__

        # ---- Location lookup ----
        if intent == Intent.LOCATION_LOOKUP:
            self.state = State.GENERAL_QUERY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Let me find some options near you.",
                backend_call={
                    "type": "location_search",
                    "params": {}
                },
                raw_debug=debug
            ).__dict__

        # ---- General knowledge ----
        if intent == Intent.GENERAL_KNOWLEDGE_QUESTION:
            self.state = State.GENERAL_QUERY
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="Let me think about that.",
                backend_call={
                    "type": "general_knowledge",
                    "params": {
                        "query": text
                    }
                },
                raw_debug=debug
            ).__dict__

        # ---- Small talk ----
        if intent == Intent.SMALL_TALK:
            self.state = State.SMALL_TALK
            return FSMResult(
                intent=intent,
                next_state=self.state,
                spoken_reply="I'm doing great, thanks for asking!",
                backend_call=None,
                raw_debug=debug
            ).__dict__

        # ---- Unknown intent ----
        self.state = State.UNKNOWN
        return FSMResult(
            intent=Intent.UNKNOWN,
            next_state=self.state,
            spoken_reply="I'm not completely sure, but I'm here to help.",
            backend_call=None,
            raw_debug=debug
        ).__dict__
