# main.py

from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse

app = FastAPI()


@app.get("/")
async def root():
    """
    Simple root endpoint so you can see something in the browser.
    """
    return JSONResponse({"message": "Vozlia backend is running"})


@app.get("/health")
async def health():
    """
    Health check endpoint for Render / uptime checks.
    """
    return {"status": "ok"}


@app.post("/twilio/inbound")
async def twilio_inbound(
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallSid: str = Form(default=""),
):
    """
    Primary Twilio Voice webhook.
    Twilio will POST here when a call comes in.

    For now, this just returns a simple TwiML message so we can confirm
    the full call path works.
    """
    # Simple TwiML response – Twilio requires valid XML with <Response> as root
    twiml = f"""
    <Response>
        <Say voice="alice">
            Hi, this is your Vozlia A I test line. Your call is working.
        </Say>
    </Response>
    """

    # Return as XML (TwiML)
    return Response(content=twiml.strip(), media_type="application/xml")


# Optional: if you run locally with `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
