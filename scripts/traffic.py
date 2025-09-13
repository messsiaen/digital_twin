# traffic.py
"""
Simple bit-queue traffic model for conversational speech.
- Queue is tracked in kilobits (kbits)
- Arrivals are produced by chosen codec bitrate (kbps) including protocol overheads
- Service is provided by effective capacity after BLER (kbps)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BitQueue:
    queue_kbits: float = 0.0
    cap_floor_kbps: float = 1e-3  # avoid division by zero

    def step(self, arrival_kbps: float, service_kbps: float, slot_seconds: float) -> dict:
        """
        Update queue for one slot.

        Returns:
            dict with {
              'queue_kbits', 'depart_kbits', 'arrive_kbits', 'queue_ms'
            }
        """
        arrive_kbits = max(0.0, float(arrival_kbps) * slot_seconds)
        serve_kbits = max(0.0, float(service_kbps) * slot_seconds)

        depart_kbits = min(self.queue_kbits + arrive_kbits, serve_kbits)
        self.queue_kbits = max(0.0, self.queue_kbits + arrive_kbits - depart_kbits)

        # Approximate queueing delay (ms) via Little's law: Q(bits)/service_rate(bits/s)
        q_ms = 1000.0 * self.queue_kbits / max(service_kbps, self.cap_floor_kbps)

        return {
            "queue_kbits": self.queue_kbits,
            "depart_kbits": depart_kbits,
            "arrive_kbits": arrive_kbits,
            "queue_ms": q_ms,
        }


def demand_from_bitrate_kbps(bitrate_kbps: float, overhead_frac: float, fec_frac: float) -> float:
    """Application-layer demand including RTP/FEC overheads."""
    return float(bitrate_kbps) * (1.0 + float(overhead_frac) + float(fec_frac))
