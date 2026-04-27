# Feature: siren-env, Property 9: Ledger Hash Chain Integrity
# Validates: Requirements 5.1, 5.2, 5.6

import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings
from hypothesis import strategies as st

from env.ledger import Ledger

# Genesis hash matches the one defined in ledger.py
_GENESIS_HASH = hashlib.sha256(b"genesis").hexdigest()

# Strategy for generating a list of (action_type, incident_id, tick) tuples
entry_strategy = st.lists(
    st.tuples(
        st.sampled_from(["fast_patch", "verified_patch", "isolate_system", "escalate_human"]),
        st.from_regex(r"INC-\d{3}", fullmatch=True),
        st.integers(min_value=0, max_value=100),
    ),
    min_size=0,
    max_size=20,
)


@given(entries=entry_strategy)
@settings(max_examples=50)
def test_ledger_hash_chain_integrity(entries):
    """
    Property 9: Ledger Hash Chain Integrity

    For any sequence of add_entry calls:
    1. ledger.hash is a 64-character lowercase hex string
    2. Recomputing the chain from scratch produces the same final hash
    3. ledger.entry_count equals the number of entries added
    4. After reset(), ledger.hash equals the genesis hash and entry_count == 0
    """
    ledger = Ledger()

    # Add all entries
    for action_type, incident_id, tick in entries:
        ledger.add_entry(action_type, incident_id, tick)

    # 1. ledger.hash is a 64-character lowercase hex string
    assert isinstance(ledger.hash, str), "ledger.hash must be a string"
    assert len(ledger.hash) == 64, f"ledger.hash must be 64 chars, got {len(ledger.hash)}"
    assert ledger.hash == ledger.hash.lower(), "ledger.hash must be lowercase"
    assert all(c in "0123456789abcdef" for c in ledger.hash), "ledger.hash must be hex"

    # 2. Recompute the chain from scratch and verify it matches the stored tip
    recomputed = _GENESIS_HASH
    for action_type, incident_id, tick in entries:
        entry_data = f"{action_type}:{incident_id}:{tick}"
        recomputed = hashlib.sha256((recomputed + entry_data).encode()).hexdigest()

    assert ledger.hash == recomputed, (
        f"Stored hash {ledger.hash!r} does not match recomputed hash {recomputed!r}"
    )

    # 3. entry_count equals the number of entries added
    assert ledger.entry_count == len(entries), (
        f"entry_count {ledger.entry_count} != expected {len(entries)}"
    )

    # 4. After reset(), hash equals genesis and entry_count == 0
    ledger.reset()
    assert ledger.hash == _GENESIS_HASH, (
        f"After reset, hash should be genesis hash, got {ledger.hash!r}"
    )
    assert ledger.entry_count == 0, (
        f"After reset, entry_count should be 0, got {ledger.entry_count}"
    )
