"""Orchestration for the RGB-normalization stage: persist the artifacts, record the evidence.

`rgb_normalization.py` is the pure transform (no clock, no filesystem, no telemetry).
`durable_store.py` owns event emission. This module is the thin layer between them that decides the
*order* things happen in, which is where the guarantees actually live:

1. store the original's bytes, content-addressed;
2. emit `ImageNormalizationStarted` -- **before** transforming, so a failure still leaves the source
   identified in the durable log;
3. transform;
4. on failure: emit `ImageNormalizationFailed` and re-raise. Never return the original.
5. on success: store the normalized bytes, emit `DerivedImageArtifactCreated`, then
   `ImageNormalizationCompleted`.

Steps 2, 4 and 5 are chained by `causation_id`, so `causation_chain(events, from_event_id=...)`
walks one page's normalization as an explicit causal path rather than an inferred one
(docs/architecture/htr-event-model.md §4).

**Failure is recorded, then raised.** Every failure condition the specification requires to be
caught before export -- undecodable image, no producible RGB, invalid dimensions, unpersistable
artifact, uncomputable hash -- reaches step 4. There is no code path in this module that returns the
un-normalized original, and `normalize_and_record` has no "on error" parameter that could be used to
ask for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_VERSION,
    NormalizationError,
    NormalizationFailure,
    NormalizationFailureCategory,
    NormalizedPageArtifact,
    PageImageArtifact,
    RgbNormalizationConfig,
)
from archivetrust.htr.preprocessing.rgb_normalization import (
    build_normalized_artifact,
    normalize_page_image,
)
from archivetrust.infrastructure.storage.blob_store import ContentAddressedBlobStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RecordedNormalization:
    """Everything one `normalize_and_record` call produced, including the event ids it emitted, so a
    caller (or a test) can assert on both the artifacts and the causal edges between the events
    announcing them -- the same shape `tests/htr/persistence/_fixtures.py::RegisteredCorpus` uses."""

    source_artifact: PageImageArtifact
    normalized_artifact: NormalizedPageArtifact
    started_event_id: str
    derived_event_id: str
    completed_event_id: str


class NormalizationService:
    """Normalizes page images, persisting bytes to a `ContentAddressedBlobStore` and provenance to a
    `DurableHtrResearchStore`.

    `store` is typed loosely (any object with the four `record_normalization_*`/
    `record_derived_image_artifact` methods) for the same reason `DurableHtrResearchStore.__init__`
    types its sink loosely: a test must be able to pass an in-memory-sink store without this module
    caring, and this module must never be the thing that decides how durable the log is.
    """

    def __init__(
        self,
        *,
        blob_store: ContentAddressedBlobStore,
        store,
        config: RgbNormalizationConfig | None = None,
    ) -> None:
        self._blobs = blob_store
        self._store = store
        self._config = config or RgbNormalizationConfig()

    @property
    def config(self) -> RgbNormalizationConfig:
        return self._config

    def read_normalized_bytes(self, artifact: NormalizedPageArtifact) -> bytes:
        """Reads a stored normalized artifact's bytes back *through* the content-addressed store, so
        the read is integrity-checked against the hash the artifact claims
        (`ContentAddressedBlobStore.get_bytes` raises `BlobIntegrityError` on a mismatch).

        The export path uses this rather than an in-memory copy of the bytes it just produced: a
        package whose files came straight from memory would be identical in the happy case and
        undetectably wrong in the case that matters, where what landed in the store is not what the
        manifest is about to claim.
        """
        return self._blobs.get_bytes(
            artifact.normalized_content_hash[len("normalized_page_") :]
        )

    def normalize_and_record(
        self,
        image_bytes: bytes,
        *,
        page_id: str,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> RecordedNormalization:
        """Runs the full stage for one page image. See the module docstring for the ordering.

        Raises `NormalizationError` on any failure, *after* having recorded an
        `ImageNormalizationFailed` event for it.
        """
        config = self._config

        # 1. Persist the original, content-addressed. A store failure here is itself a recordable
        #    failure -- but it happens before `ImageNormalizationStarted`, so there is no open
        #    attempt to fail; the exception is annotated and raised, not silently retried.
        try:
            digest, _ = self._blobs.put_bytes(image_bytes)
            source_artifact = PageImageArtifact.create(
                image_bytes=image_bytes,
                page_id=page_id,
                storage_path=str(self._blobs.path_for(digest)),
            )
        except OSError as exc:
            raise NormalizationError(
                NormalizationFailure.create(
                    page_id=page_id,
                    category=NormalizationFailureCategory.ARTIFACT_NOT_PERSISTED,
                    reason=f"cannot store the source image bytes: {type(exc).__name__}: {exc}",
                    configuration_hash=config.configuration_hash,
                    occurred_at=_utc_now(),
                )
            ) from exc

        # 2. Open the attempt in the durable log, before any transformation.
        started_event_id = self._store.record_normalization_started(
            source_artifact,
            configuration_hash=config.configuration_hash,
            normalization_version=RGB_NORMALIZATION_VERSION,
            caused_by=caused_by,
            correlation_id=correlation_id,
        )

        # 3./4. Transform; record any failure against the now-open attempt, then re-raise.
        try:
            result = normalize_page_image(image_bytes, config=config)
            executed_at = _utc_now()
            normalized_hash_digest, _ = self._blobs.put_bytes(result.image_bytes)
            normalized_artifact = build_normalized_artifact(
                result,
                source_artifact=source_artifact,
                storage_path=str(self._blobs.path_for(normalized_hash_digest)),
                executed_at=executed_at,
                config=config,
            )
        except NormalizationError as exc:
            # The transform's own failure record carries no page id (it is pure and does not know
            # one); re-stamp it with the page and timestamp this layer does know, so the durable
            # record is scoped. Category and reason are the transform's, never reinterpreted here.
            failure = exc.failure.model_copy(
                update={"page_id": page_id, "occurred_at": _utc_now()}
            )
            self._store.record_normalization_failed(
                failure, caused_by=started_event_id, correlation_id=correlation_id
            )
            raise NormalizationError(failure) from exc
        except (OSError, ValueError) as exc:
            failure = NormalizationFailure.create(
                page_id=page_id,
                category=NormalizationFailureCategory.ARTIFACT_NOT_PERSISTED,
                reason=f"cannot persist the normalized artifact: {type(exc).__name__}: {exc}",
                configuration_hash=config.configuration_hash,
                occurred_at=_utc_now(),
                source_content_hash=source_artifact.content_hash,
            )
            self._store.record_normalization_failed(
                failure, caused_by=started_event_id, correlation_id=correlation_id
            )
            raise NormalizationError(failure) from exc

        # 5. Record the derived artifact, then the terminal success marker.
        derived_event_id = self._store.record_derived_image_artifact(
            normalized_artifact, caused_by=started_event_id, correlation_id=correlation_id
        )
        completed_event_id = self._store.record_normalization_completed(
            normalized_artifact, caused_by=derived_event_id, correlation_id=correlation_id
        )

        return RecordedNormalization(
            source_artifact=source_artifact,
            normalized_artifact=normalized_artifact,
            started_event_id=started_event_id,
            derived_event_id=derived_event_id,
            completed_event_id=completed_event_id,
        )

    def normalize_file_and_record(
        self,
        image_path: Path | str,
        *,
        page_id: str,
        caused_by: str | None = None,
        correlation_id: str | None = None,
    ) -> RecordedNormalization:
        """Convenience wrapper reading bytes from disk. An unreadable path is an
        `UNDECODABLE_IMAGE` failure recorded the same way as undecodable bytes -- a missing file and
        a corrupt file are equally "this page could not be normalized", and neither may result in an
        un-normalized original being packaged."""
        path = Path(image_path)
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            raise NormalizationError(
                NormalizationFailure.create(
                    page_id=page_id,
                    category=NormalizationFailureCategory.UNDECODABLE_IMAGE,
                    reason=f"cannot read {path}: {type(exc).__name__}: {exc}",
                    configuration_hash=self._config.configuration_hash,
                    occurred_at=_utc_now(),
                )
            ) from exc
        return self.normalize_and_record(
            image_bytes, page_id=page_id, caused_by=caused_by, correlation_id=correlation_id
        )
