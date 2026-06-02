# 
# Encapsulates the violation state machine for each tracked vehicle.
#
# Logic summary:
#   - A vehicle must remain in an unauthorized lane CONTINUOUSLY
#     for VIOLATION_SECONDS_THRESHOLD seconds before it is flagged.
#   - If it exits before the timer completes, the entry time resets
#     (forgiveness logic).
#   - Once flagged, the vehicle stays in "violation" state for the
#     rest of the video session.
#
# Possible statuses returned by .update():
#   "safe"      → vehicle is in a normal lane
#   "warning"   → vehicle entered unauthorized lane, timer running
#   "violation" → threshold exceeded, logged to DB
# 

import time

class ViolationManager:
    """Tracks per-vehicle violation state using real-time elapsed seconds.

    Args:
        threshold_seconds (float): How long a vehicle must stay in an
                                   unauthorized lane to trigger a violation.
    """

    def __init__(self, threshold_seconds: float):
        self.threshold_seconds = threshold_seconds

        # Per track_id: wall-clock time when vehicle first entered unauthorized lane
        self._entry_time: dict[int, float] = {}

        # Set of track_ids that have already been fully flagged as violations
        self._violated: set[int] = set()

    def update(self, track_id: int, in_unauthorized: bool) -> str:
        """Update the violation state for a single vehicle this frame.

        Returns:
            "safe"      — vehicle is compliant
            "warning"   — vehicle is in unauthorized lane, timer running
            "violation" — vehicle exceeded threshold (first time or already logged)
        """
        if track_id in self._violated:
            return "violation"

        if in_unauthorized:
            if track_id not in self._entry_time:
                self._entry_time[track_id] = time.time()

            elapsed = time.time() - self._entry_time[track_id]
            if elapsed >= self.threshold_seconds:
                self._violated.add(track_id)
                return "violation"

            return "warning"
        else:
            self._entry_time.pop(track_id, None)
            return "safe"

    def already_violated(self, track_id: int) -> bool:
        """True if this vehicle has already been recorded as a violator."""
        return track_id in self._violated
