"""
feedback_store.py
─────────────────
All Firestore read / write for the feedback feature lives here.
app.py imports only: save_feedback()  and  get_feedback_stats()

Collection layout (single collection, no sub-collections):
  feedback/
    {auto-id}  →  { rating: int, text: str, created_at: timestamp }

LOCAL DEV FALLBACK
  If google-cloud-firestore is not installed, or if the Firestore client
  can't initialise (no credentials), the module silently falls back to an
  in-memory list.  Writes survive the process lifetime so you can POST a
  rating and immediately GET stats back — but obviously nothing persists
  across restarts.  On Cloud Run the real Firestore path is always used.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── try to get a real Firestore client; fall back to None ──────────
_db = None
_USE_FIRESTORE = False

try:
    from google.cloud import firestore
    _db = firestore.Client()
    _USE_FIRESTORE = True
    logger.info("✅ Firestore client initialised (real)")
except Exception as e:
    logger.warning(
        f"⚠️  Firestore not available ({type(e).__name__}: {e}) — "
        "using in-memory fallback.  Feedback will not persist across restarts."
    )

# ── in-memory fallback store ───────────────────────────────────────
_mem: list = []

COLLECTION = "feedback"


# ── public API ──────────────────────────────────────────────────────

async def save_feedback(rating: int, text: str) -> dict:
    """
    Persist one feedback entry.
    Returns a dict with an 'id' so the caller can confirm.
    """
    if not (1 <= rating <= 5):
        raise ValueError("rating must be 1-5")

    if _USE_FIRESTORE:
        doc = _db.collection(COLLECTION).document()   # auto-generated id
        doc.set({
            "rating":     rating,
            "text":       text.strip() if text else "",
            "created_at": datetime.now(timezone.utc),
        })
        logger.info(f"💬 Feedback saved (Firestore)  id={doc.id}  rating={rating}")
        return {"id": doc.id}
    else:
        # in-memory path
        entry = {
            "rating":     rating,
            "text":       text.strip() if text else "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _mem.append(entry)
        logger.info(f"💬 Feedback saved (memory)  index={len(_mem)-1}  rating={rating}")
        return {"id": str(len(_mem) - 1)}


async def get_feedback_stats() -> dict:
    """
    Return { "avg": float, "count": int }.
    Reads from Firestore when available, otherwise from the in-memory list.
    """
    if _USE_FIRESTORE:
        stream = _db.collection(COLLECTION).stream()
        total  = 0
        count  = 0
        for doc in stream:
            rating = doc.to_dict().get("rating")
            if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                total += rating
                count += 1
    else:
        total = sum(e["rating"] for e in _mem)
        count = len(_mem)

    avg = round(total / count, 1) if count else 0.0
    source = "Firestore" if _USE_FIRESTORE else "memory"
    logger.info(f"📊 Feedback stats ({source})  avg={avg}  count={count}")
    return {"avg": avg, "count": count}