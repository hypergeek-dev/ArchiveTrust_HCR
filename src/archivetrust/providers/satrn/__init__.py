"""SATRN (`Riksarkivet/satrn_htr`) `HtrMethodAdapter` implementation -- see `adapter.py`'s module
docstring for the interface investigation and the isolated-subprocess architecture, and
`README.md` for install/setup instructions and measured real-inference behavior.
"""

from __future__ import annotations

from archivetrust.providers.satrn.adapter import (
    ADAPTER_VERSION,
    DEFAULT_MODEL_ID,
    METHOD_ID,
    SatrnAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)

__all__ = [
    "ADAPTER_VERSION",
    "DEFAULT_MODEL_ID",
    "METHOD_ID",
    "SatrnAdapter",
    "build_evidence",
    "build_failure_record",
    "build_observation_payloads",
    "normalize_transcription",
]
