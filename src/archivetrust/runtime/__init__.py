"""Provider Runtime Abstraction & Model Management — infrastructure only.

**Architectural objective** (this milestone): make ArchiveTrust independent of any specific
inference runtime, the same way it is already independent of any specific document-understanding
method. Two responsibilities that were previously conflated in a single injected callable
(`providers.qwen_vl.backends.LocalInferenceRunner`) are now explicit and separate:

- A **Provider** (`providers/qwen_vl/`, `providers/docling/`, ...) performs document understanding
  — it maps provider-native output into Evidence and Observations (ROADMAP.md S5.5, S5.7).
- An **Inference Runtime** (this package) executes a model. It knows nothing about documents,
  Evidence, Observations, or the Canonical Observation Ontology — only "run this model on this
  input, return this output."
- The **Model Registry** (`model_registry.py`) resolves a logical identifier (e.g. "Primary Vision
  Provider") to a concrete runtime + model + revision + device + cache location. Providers request
  inference through a runtime; they never resolve model paths, choose a runtime implementation, or
  know a caching/download strategy themselves.

```
Vision Provider (providers/qwen_vl/)
        │  requests inference via the existing QwenVLBackend/LocalInferenceRunner seam
        ▼
Inference Runtime (this package: TransformersRuntime, OpenAICompatibleRuntime, ...)
        │  executes a resolved model
        ▼
External Model (models/ directory — never compiled into the executable)
```

**Explicit non-goals, honored:** this package never imports from `archivetrust.domain`,
`archivetrust.providers`, `archivetrust.application`, `archivetrust.learning`, or `archivetrust.review`.
It has no opinion about Evidence, Observations, telemetry events, comparison, alignment, or
confidence. The one place it touches the rest of the system is a single bridge module
(`providers/qwen_vl/runtime_backend.py`) that adapts an `InferenceRuntime` to the *existing*,
*unchanged* `LocalInferenceRunner` Protocol — the seam Milestone 3 already left for exactly this
purpose (Guiding Principle 9). No provider adapter, domain type, or telemetry event changes.

**Models are treated exactly like archive objects: immutable external resources ArchiveTrust
consumes, never owns or embeds.** They live in a `models/` directory alongside the executable
(`deployment_layout.py`), never compiled in; the same executable supports Qwen2.5-VL, GLM-OCR,
PaddleOCR-VL, or a future VLM through configuration alone.
"""
