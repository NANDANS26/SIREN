"""SHA-256 proof ledger for the SIREN RL environment."""
from __future__ import annotations

import hashlib
from typing import List, Dict, Any

# Genesis hash: sha256(b"genesis")
_GENESIS_HASH: str = hashlib.sha256(b"genesis").hexdigest()


class Ledger:
    """Maintains an in-memory SHA-256 hash chain of verified actions.

    Each entry records the action type, incident id, tick, and a hash
    chained from the previous entry. Zero network calls are made.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._current_hash: str = _GENESIS_HASH

    def add_entry(self, action_type: str, incident_id: str, tick: int) -> None:
        """Append a new entry and advance the hash chain.

        The new hash is: sha256(previous_hash + f"{action_type}:{incident_id}:{tick}")
        where the concatenation is done as strings then encoded to bytes.
        """
        entry_data = f"{action_type}:{incident_id}:{tick}"
        new_hash = hashlib.sha256(
            (self._current_hash + entry_data).encode()
        ).hexdigest()
        self._entries.append({
            "action_type": action_type,
            "incident_id": incident_id,
            "tick": tick,
            "hash": new_hash,
        })
        self._current_hash = new_hash

    def reset(self) -> None:
        """Clear all entries and restore the genesis hash."""
        self._entries = []
        self._current_hash = _GENESIS_HASH

    @property
    def hash(self) -> str:
        """Return the current chain tip as a 64-character lowercase hex string."""
        return self._current_hash

    @property
    def entry_count(self) -> int:
        """Return the number of entries in the ledger."""
        return len(self._entries)
