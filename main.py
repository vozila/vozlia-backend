# main.py

import json
import logging

from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import Response, JSONResponse

from openai import OpenAI

# ========= Logging Setup =========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")

# ========= OpenAI Client =========
# Requires OPENAI_API_KEY in the environment (e.g. in Render)
client = OpenAI()

app = FastAPI()


# ========= GPT Helper =========
def generate_gpt_reply(user_text: str) -> str:
    """
    Call GPT from the Vozlia backend.
    This is a simple one-shot chat completion for now.
    """
    if not user_text:
        return "I didn't receive any text to respond to."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or gpt-4o, depending on your access
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Vozlia, a friendly but efficient AI voice assistant. "
                        "You answer as if you're speaking on a phone call: natural, concise, and helpful."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.exception(f"Error calling OpenAI: {e}")
        return "I'm having trouble connecting to my brain right now. Please try again later."


# ========= Basic Endpoints =========
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
    """
    return {"status": "ok"}


# ========= Debug GPT Endpoint (GET for simplicity) =========
@app.get("/debug/gpt")
async def debug_gpt(text: str = Query(default="")):
    """
    Simple debugging endpoint to check GPT connectivity from the backend.

    Example:
      curl "https://vozlia-backend.onrender.com/debug/gpt?text=Hello+Vozlia"
    """
    logger.info(f"/debug/gpt called with text={text!r}")
    reply = generate_gpt_reply(text)
    return {"reply": reply}


# ========= Twilio Voice Webhook =========
@app.post("/twilio/inbound")
async def twilio_inbound(
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallSid: str = Form(default="")
):
    """
    Twilio hits this when a call comes in.
    We greet the caller, then hand off the audio stream to /twilio/stream
    using <Connect><Stream>.
    """

    logger.info(f"Incoming call: From={From}, To={To}, CallSid={CallSid}")

    # IMPORTANT: Twilio requires a wss:// URL for streaming
    stream_url = "wss://vozlia-backend.onrender.com/twilio/stream"

    twiml = f"""
    <Response>
        <Say voice="alice">
            Hi, this is your Vozlia A I assistant. I am connecting our secure audio stream now.
        </Say>
        <Connect>
            <Stream url="{stream_url}" />
        </Connect>
    </Response>
    """

    return Response(content=twiml.strip(), media_type="application/xml")


# ========= Twilio Media Stream WebSocket =========
@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    For now, we just log events.
    """

    await ws.accept()
    logger.info("Twilio stream connected")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            event_type = msg.get("event", "unknown")
            logger.info(f"Twilio stream event: {event_type}")

            if event_type == "start":
                logger.info(f"Stream start: {msg}")

            elif event_type == "media":
                # Audio frame in msg["media"]["payload"] (base64)
                # payload = msg.get("media", {}).get("payload", "")
                # logger.info(f"Media frame received, payload length={len(payload)}")
                pass

            elif event_type == "stop":
                logger.info("Stream stop event received. Closing WebSocket.")
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception(f"Error in Twilio stream: {e}")
    finally:
        await ws.close()
        logger.info("Twilio stream closed")


# ========= Local Dev Entrypoint =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
