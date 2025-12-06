# main.py
# main.py

from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse

app = FastAPI()


def generate_ai_reply(speech: str, from_number: str = "", to_number: str = "") -> str:
    """
    Placeholder AI logic for Vozlia.

    Right now this is simple rule-based behavior so you can test the flow end-to-end.
    Later, you'll replace this with a real LLM call (OpenAI, etc.).
    """

    if not speech:
        return (
            "I didn't quite catch that. "
            "In the future, I will be able to help you with scheduling, messages, and other tasks."
        )

    # Very basic intent-feel behavior
    lower_speech = speech.lower()

    if "appointment" in lower_speech or "schedule" in lower_speech:
        return (
            "It sounds like you want to schedule something. "
            "In the production version, I will check your calendar and suggest available times."
        )

    if "message" in lower_speech or "tell" in lower_speech:
        return (
            "You want to leave a message. "
            "In the future, I will record this for the subscriber and send them a summary by text."
        )

    if "who is this" in lower_speech or "what is this" in lower_speech:
        return (
            "I am the Vozlia voice assistant, here to screen calls, take messages, "
            "and help with personal and business tasks 24 hours a day."
        )

    # Default generic helpful reply
    return (
        f"You said: {speech}. "
        "Thank you for calling the Vozlia assistant. "
        "Soon I will be able to act on this, like booking appointments or sending messages."
    )


@app.get("/")
async def root():
    """
    Basic root endpoint to verify deployment.
    """
    return JSONResponse({"message": "Vozlia backend is running"})


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Render, uptime monitors, and your browser can use this.
    """
    return {"status": "ok"}


@app.post("/twilio/inbound")
async def twilio_inbound(
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallSid: str = Form(default="")
):
    """
    First endpoint Twilio calls when someone dials your number.
    Greets the caller and starts collecting speech using <Gather>.

    After the caller speaks, Twilio will POST the transcript to /twilio/continue.
    """

    # IMPORTANT: Twilio needs a full absolute URL here for the action
    action_url = "https://vozlia-backend.onrender.com/twilio/continue"

    twiml = f"""
    <Response>
        <Say voice="alice">
            Hi, this is your Vozlia A I assistant. How can I help you today?
        </Say>

        <Gather input="speech" action="{action_url}" method="POST" timeout="5">
            <Say voice="alice">
                Please tell me how I can assist you, then pause for a moment.
            </Say>
        </Gather>

        <!-- Fallback if no speech was detected -->
        <Say voice="alice">
            I did not hear anything. Please call again when you're ready.
        </Say>
        <Hangup/>
    </Response>
    """

    return Response(content=twiml.strip(), media_type="application/xml")


@app.post("/twilio/continue")
async def twilio_continue(
    SpeechResult: str = Form(default=""),
    Fro
::contentReference[oaicite:0]{index=0}
0.0.0.0", port=8000, reload=True)
