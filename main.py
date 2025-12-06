# main.py

from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse

app = FastAPI()


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
    Greet the caller and start collecting speech using <Gather>.
    After the caller speaks, Twilio will POST the transcript to /twilio/continue.
    """

    # IMPORTANT: Twilio needs a full absolute URL here.
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
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallSid: str = Form(default="")
):
    """
    Twilio posts the transcribed speech here after the user talks.
    For now, we simply repeat back what the caller said.
    Later, this is where your real AI logic will live.
    """

    if SpeechResult:
        reply_text = f"You said: {SpeechResult}. Thanks for speaking with the Vozlia assistant."
    else:
        reply_text = "I did not catch that. Let's try again another time."

    twiml = f"""
    <Response>
        <Say voice="alice">{reply_text}</Say>
        <Hangup/>
    </Response>
    """

    return Response(content=twiml.strip(), media_type="application/xml")


# Optional: Local dev entrypoint for running `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
