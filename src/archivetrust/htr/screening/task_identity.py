"""Stable, deterministic identity for one unit of work in a reliability run.

The whole resume mechanism rests on one property: **the same task, computed in a different process
on a different day, must produce the same identifier** -- and a task under any changed condition
must produce a different one. That is what makes "have I already done this?" a question the durable
telemetry log can answer, rather than a question a side-car state file has to be trusted about.

## The tuple

    (experiment_version_id, method_id, model_revision, page_or_crop_id, input_hash,
     preprocessing_version, sampling_version, configuration_hash)

Each field is load-bearing and none is decorative:

* `experiment_version_id` / `configuration_hash` -- the run's whole pinned configuration. A changed
  threshold, a changed sample, a changed crops-per-page and the identity changes, so previously
  completed work is correctly *not* adopted into a differently-configured run.
* `method_id` + `model_revision` -- SATRN and Florence-2 are independent tasks over the same crop,
  which is what lets them resume independently; a swapped checkpoint is a different task, never a
  silent continuation.
* `page_or_crop_id` -- *which* input. For a segmentation task this is the `page_id`. For a
  recognition task it is the deterministic slot `"<page_id>#crop<index>"`, where `index` is the
  position in the page's reading-ordered detected lines. The `InputCrop.crop_id` itself is
  `new_id()`-random (`htr/corpus/models.py`) and so would make identity depend on when segmentation
  happened to run; the real crop id travels on the `MethodRun` instead, where it belongs.
* `input_hash` -- the content address of the bytes actually fed to the method
  (`InputCrop.hash` for a crop, the inventory's page content hash for a page). This is what makes
  the identity a claim about *content*, not about a filename.
* `preprocessing_version` / `sampling_version` -- the two versioned upstream stages whose output the
  task consumes.

`task_key` is a sha256 over the canonical JSON of exactly those eight fields.

## How completion is recorded

Two derived identifiers, both deterministic functions of `task_key`:

* `segmentation_run_id` -> carried by `SegmentationRunCompleted`;
* `method_run_id` -> carried by `MethodRunStarted` / `MethodRunCompleted`.

A task is complete **iff** its terminal event is present in the durable log with that id. Nothing
else counts -- not a progress cache, not a file on disk, not a partially populated projection.

## Why this cannot mark an interrupted task complete

Neither derived event is appended before the work happens. A recognition task appends nothing at all
until `adapter.recognize()` has genuinely returned, because the `MethodRun` record cannot even be
constructed before then (`MethodRun.outcome` and `MethodRun.evidence_id` are required fields whose
values are the call's result). So an interruption at any point during the call leaves the log with
no completion marker for that `method_run_id`, and the next pass re-executes it. The guarantee is
structural, not a matter of ordering discipline that a future edit could quietly reverse.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict


class TaskIdentity(BaseModel):
    """One task's stable identity. Frozen: a task cannot be re-labelled once computed."""

    model_config = ConfigDict(frozen=True)

    experiment_version_id: str
    method_id: str
    model_revision: str
    page_or_crop_id: str
    input_hash: str
    preprocessing_version: str
    sampling_version: str
    configuration_hash: str

    def identity_tuple(self) -> tuple[str, ...]:
        """The tuple, in the order the specification states it -- returned so a test can assert on
        the ordering rather than on a hash whose inputs it cannot see."""
        return (
            self.experiment_version_id,
            self.method_id,
            self.model_revision,
            self.page_or_crop_id,
            self.input_hash,
            self.preprocessing_version,
            self.sampling_version,
            self.configuration_hash,
        )

    @property
    def task_key(self) -> str:
        payload = json.dumps(
            {
                "experiment_version_id": self.experiment_version_id,
                "method_id": self.method_id,
                "model_revision": self.model_revision,
                "page_or_crop_id": self.page_or_crop_id,
                "input_hash": self.input_hash,
                "preprocessing_version": self.preprocessing_version,
                "sampling_version": self.sampling_version,
                "configuration_hash": self.configuration_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "task_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def method_run_id(self) -> str:
        """The deterministic `MethodRun` id whose `MethodRunCompleted` marks this task done."""
        return "method_run_" + self.task_key.split("_", 1)[1]

    @property
    def segmentation_run_id(self) -> str:
        """The deterministic segmentation-run id whose `SegmentationRunCompleted` marks a page's
        segmentation done."""
        return "segmentation_run_" + self.task_key.split("_", 1)[1]


def crop_slot_id(page_id: str, index: int) -> str:
    """The deterministic `page_or_crop_id` for the `index`-th detected line of a page.

    Not the `InputCrop.crop_id` -- see the module docstring for why identity must not depend on a
    randomly minted id.
    """
    return f"{page_id}#crop{index}"
