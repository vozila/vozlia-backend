# Known Good Baseline (Flow B Realtime)

Status: ✅ Twilio + Realtime working end-to-end.

Evidence:
- /twilio/inbound returns TwiML and Twilio connects to /twilio/stream
- Realtime connects and sends session.update + greeting
- VAD + transcripts working
- FSM email intent triggers Gmail summary and returns spoken_reply

Rollback point: this commit
