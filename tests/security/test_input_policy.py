from __future__ import annotations

import pytest

from archivetrust.security.input_policy import (
    HostileInputRejected,
    InputSecurityPolicy,
    normalized_filename,
    validate_source_file,
)
from tests.infrastructure.rendering._minimal_pdf import write_minimal_pdf


def test_normalizes_untrusted_filename_and_drops_path_components() -> None:
    assert normalized_filename(r"..\..\Council:Minutes?.pdf") == "Council_Minutes_.pdf"
    with pytest.raises(HostileInputRejected):
        normalized_filename("CON.pdf")


def test_admits_real_pdf_and_rejects_extension_spoof(tmp_path) -> None:
    real_pdf = tmp_path / "minutes.pdf"
    write_minimal_pdf(real_pdf, num_pages=1, width=300, height=400)
    policy = InputSecurityPolicy()
    assert validate_source_file(real_pdf, original_filename=real_pdf.name, policy=policy) == "minutes.pdf"

    spoof = tmp_path / "spoof.pdf"
    spoof.write_bytes(b"this is not a PDF")
    with pytest.raises(HostileInputRejected, match="does not match"):
        validate_source_file(spoof, original_filename=spoof.name, policy=policy)


def test_rejects_oversized_and_embedded_file_pdf(tmp_path) -> None:
    oversized = tmp_path / "large.pdf"
    oversized.write_bytes(b"%PDF-1.7\n123456789")
    with pytest.raises(HostileInputRejected, match="outside the allowed range"):
        validate_source_file(
            oversized,
            original_filename=oversized.name,
            policy=InputSecurityPolicy(max_file_bytes=10),
        )

    embedded = tmp_path / "embedded.pdf"
    embedded.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /EmbeddedFile >>")
    with pytest.raises(HostileInputRejected, match="embedded files"):
        validate_source_file(embedded, original_filename=embedded.name, policy=InputSecurityPolicy())
