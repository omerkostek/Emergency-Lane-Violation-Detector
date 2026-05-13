# core/violation_manager.py
# ─────────────────────────────────────────────────────────────
# Encapsulates the violation state machine for each tracked vehicle.
#
# Logic summary:
#   - A vehicle must remain in an unauthorized lane CONTINUOUSLY
#     for VIOLATION_SECONDS_THRESHOLD seconds before it is flagged.
#   - If it exits before the timer completes, the counter resets
#     (forgiveness logic — mirrors real EDS camera behavior).
#   - Once flagged, the vehicle stays in "violation" state for the
#     rest of the video session.
#
# Possible statuses returned by .update():
#   "safe"      → vehicle is in a normal lane
#   "warning"   → vehicle entered unauthorized lane, timer running
#   "violation" → threshold exceeded, logged to DB
# ─────────────────────────────────────────────────────────────


class ViolationManager:
    """Tracks per-vehicle violation state and pending frame counts.
    
    Args:
        fps (float): Frames per second of the source video, used to
                     convert the seconds threshold into a frame count.
        threshold_seconds (float): How long a vehicle must stay in an
                                   unauthorized lane to trigger a violation.
    """

    def __init__(self, fps: float, threshold_seconds: float):
        # Convert time threshold to a frame count once at startup
        self.threshold_frames = int(fps * threshold_seconds)

        # Per track_id: how many consecutive frames in unauthorized lane
        self._pending: dict[int, int] = {}

        # Set of track_ids that have already been fully flagged as violations
        self._violated: set[int] = set()

    def update(self, track_id: int, in_unauthorized: bool) -> str:
        """Update the violation state for a single vehicle this frame.
        
        Returns:
            "safe"      — vehicle is compliant
            "warning"   — vehicle is in unauthorized lane, timer running
            "violation" — vehicle exceeded threshold (first time or already logged)
        """
        # Once a vehicle has been flagged it stays flagged for the session
        if track_id in self._violated:
            return "violation"

        if in_unauthorized:
            # Increment the consecutive-frame counter
            self._pending[track_id] = self._pending.get(track_id, 0) + 1

            if self._pending[track_id] >= self.threshold_frames:
                # Threshold crossed — promote to full violation
                self._violated.add(track_id)
                return "violation"

            return "warning"
        else:
            # Vehicle left the unauthorized zone — forgive and reset counter
            self._pending[track_id] = 0
            return "safe"

    def is_new_violation(self, track_id: int) -> bool:
        """True only on the exact frame the violation threshold was first crossed."""
        return (
            track_id in self._violated
            and self._pending.get(track_id, 0) == self.threshold_frames
        )

    def already_violated(self, track_id: int) -> bool:
        """True if this vehicle has already been recorded as a violator."""
        return track_id in self._violated
