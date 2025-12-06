# main.py

import json
import logging

from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse

# Basic logging setup so you can see stream events in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")

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


@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.

    Twilio will send JSON messages like:
      - event: "start"
      - event: "media"  (with base64-encoded audio payload)
      - event: "stop"

    For now, we just accept the stream and log the events.
    Later, this is where you'll bridge audio to/from OpenAI Realtime.
    """

    await ws.accept()
    logger.info("Twilio stream connected")

    try:
        while True:
            # Twilio sends JSON text frames
            raw = await ws.receive_text()
            msg = json.loads(raw)

            event_type = msg.get("event", "unknown")
            logger.info(f"Twilio stream event: {event_type}")

            if event_type == "start":
                logger.info(f"Stream start: {msg}")
            elif event_type == "media":
                # This contains audio in base64 at msg["media"]["payload"]
                # We won't decode yet; just log that we received media
                # payload_len = len(msg.get("media", {}).get("payload", ""))
                # logger.info(f"Media frame received, payload length={payload_len}")
                pass
            elif event_type == "stop":
                logger.info("Stream stop event received. Closing WebSocket.")
                break

            # OPTIONAL: you could send marks or other messages back to Twilio if needed.
            # For now, we don't send anything back.

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception(f"Error in Twilio stream: {e}")
    finally:
        await ws.close()
        logger.info("Twilio stream closed")


# Optional: Local dev entrypoint for running `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
