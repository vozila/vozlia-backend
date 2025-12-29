Last good known update Dec 29 3:30PM
still working on portal
completed the first iteration of short-term  memory


Nice — that confirms caller-scoped Postgres TTL cache is working (across calls), which is the hardest part.

What “long-term memory for all skills” means in practice

You already have:

Session memory (per call) ✅

Caller TTL cache (cross-call, per caller) ✅

Next is durable memory (not TTL-cached results), so any skill can:

read a compact “what we know about this caller” context

write back important outcomes for future calls

And you want it per-tenant toggleable (env vars for now; later portal settings).
