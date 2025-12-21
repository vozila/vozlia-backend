# api/routers/twilio.py
from fastapi import APIRouter
from vozlia_twilio.inbound import router as twilio_inbound_router
from vozlia_twilio.stream import twilio_stream

router = APIRouter()
router.include_router(twilio_inbound_router)
# api/routers/twilio.py (continued)
def mount_twilio_ws(app):
    app.add_api_websocket_route("/twilio/stream", twilio_stream)

# websocket route must be added on the FastAPI app, but we can expose it here
# If you prefer: keep this in main.py using app.add_api_websocket_route.
# FastAPI doesn't allow APIRouter.add_api_websocket_route directly prior to some versions,
# so safest is: do it in main.py OR export a function.
