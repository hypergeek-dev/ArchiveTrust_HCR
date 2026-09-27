"""`htr/benchmark/` -- the independent, frozen-model HTR benchmark harness (docs/BENCHMARK_PROTOCOL.md).

Answers one question fairly: how do two *frozen* line recognizers (the scratch-trained Loghi model,
Experiment 2 epoch 7, and Riksarkivet's Swedish Lion Libre) transcribe exactly the same unseen,
ground-truthed line images?

Stages, each a separate command so every intermediate artifact can be inspected:

    incoming/<source>  --inspect-->  work/<source>/inspection/      (never writes into incoming/)
                       --build---->  work/<source>/candidate/       (canonical lines + review queue)
                       --freeze--->  benchmark/<benchmark_id>/      (immutable manifest + hashes)
                       --run------>  reports/<run_id>/predictions/  (one JSONL per model, no scoring)
                       --score---->  reports/<run_id>/              (metrics, pairwise analysis, report)

Not to be confused with the pre-HTR `archivetrust.benchmark` package, which measures telemetry the
old document-trust pipeline recorded and has nothing to do with line-level recognition accuracy.
"""
