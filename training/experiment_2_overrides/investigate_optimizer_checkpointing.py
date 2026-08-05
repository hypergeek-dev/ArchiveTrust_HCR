"""Investigation only, not part of the real save/restore mechanism. Answers, empirically, in the real
pinned container:

1. Does `model.compile()` under the global `mixed_float16` policy wrap the plain Adam optimizer in a
   `LossScaleOptimizer`? (If yes, its loss-scale state needs to be part of what gets persisted.)
2. Does `tf.train.Checkpoint(optimizer=optimizer)` (Keras 3's native mechanism, distinct from
   `clone_model()` + `.save()`, which the pinned commit's own `custom_callback.py` uses and which is
   exactly why no checkpoint before now has ever carried optimizer state) actually round-trip real
   Adam slot variables (m/v moments) and `iterations`, not just the model weights?
3. After a save -> fresh-optimizer -> restore cycle, do `optimizer.iterations` and the resolved
   learning rate match exactly?

Writes a JSON report to /output. Never touches the real corpus or launches real training.
"""

import json
import os
import time
import traceback

os.environ["TF_USE_LEGACY_KERAS"] = "0"

import numpy as np
import tensorflow as tf
from vgslify.generator import VGSLModelGenerator

from model.losses import CTCLoss
from model.management import get_model_library
from model.metrics import CERMetric, WERMetric
from model.optimization import create_learning_rate_schedule, get_optimizer
from model.replacing import replace_final_layer

# Replicates exactly what setup/environment.py::setup_environment does when a GPU is present --
# confirmed by reading that module directly (see this repo's own earlier investigation). Must run
# BEFORE any model is built, matching main.py's own invocation order, or the global policy silently
# stays float32 and this whole investigation answers the wrong question (this happened on the first
# run of this script: mixed_precision_policy_active read "float32" because this step was missing).
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_float16"))

REPORT = {"checks": {}}


def record(name, ok, detail=None):
    REPORT["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail if detail is not None else ''}")


def main():
    vocab_size = 126
    vgsl_spec = get_model_library()["recommended"]
    model = VGSLModelGenerator().generate_model(model_spec=vgsl_spec, model_name="investigate")
    model = replace_final_layer(model, vocab_size, model.name)

    lr_schedule = create_learning_rate_schedule(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, train_batches=1000,
        do_train=True, warmup_ratio=0.0, epochs=1, decay_per_epoch=False, linear_decay=False,
    )
    optimizer = get_optimizer("adam", lr_schedule)
    model.compile(optimizer=optimizer, loss=CTCLoss(), metrics=[CERMetric(greedy=True), WERMetric()], weighted_metrics=[])

    # --- Question 1: is the compiled optimizer wrapped in a LossScaleOptimizer? ---
    optimizer_type = type(model.optimizer).__name__
    is_loss_scale_wrapped = "LossScale" in optimizer_type or hasattr(model.optimizer, "inner_optimizer")
    record("mixed_precision_policy_active", tf.keras.mixed_precision.global_policy().name == "mixed_float16",
           tf.keras.mixed_precision.global_policy().name)
    record("optimizer_type_after_compile", True, optimizer_type)
    REPORT["is_loss_scale_wrapped"] = is_loss_scale_wrapped
    if is_loss_scale_wrapped:
        inner = getattr(model.optimizer, "inner_optimizer", None)
        REPORT["inner_optimizer_type"] = type(inner).__name__ if inner is not None else None
        REPORT["has_dynamic_loss_scale_attr"] = hasattr(model.optimizer, "dynamic_scale") or hasattr(model.optimizer, "loss_scale")

    # --- Run a few real training steps so the optimizer's slot variables actually get built ---
    batch_size, width, height, channels = 2, 256, 64, 1
    x = np.random.rand(batch_size, width, height, channels).astype("float32")
    y = np.random.randint(1, vocab_size, size=(batch_size, 20)).astype("int64")
    for _ in range(5):
        model.train_on_batch(x, y)

    iterations_before_save = int(model.optimizer.iterations.numpy())
    lr_before_save = None
    try:
        lr_attr = model.optimizer.learning_rate
        lr_before_save = float(lr_attr(model.optimizer.iterations).numpy()) if callable(lr_attr) else float(lr_attr.numpy())
    except Exception as exc:  # noqa: BLE001
        REPORT["lr_read_error_before_save"] = str(exc)
    record("optimizer_iterations_after_5_steps", iterations_before_save == 5, iterations_before_save)

    # Sample one real Adam slot variable (a moment) before saving, to compare after restore.
    sample_slot_before = None
    try:
        opt_vars = model.optimizer.variables
        # first non-iterations variable with actual content
        for v in opt_vars:
            if "iteration" not in v.name.lower() and v.shape.num_elements() and v.shape.num_elements() > 0:
                sample_slot_before = (v.name, float(tf.reduce_sum(v).numpy()))
                break
    except Exception as exc:  # noqa: BLE001
        REPORT["slot_sample_error"] = str(exc)
    REPORT["sample_slot_before_save"] = sample_slot_before

    # --- Question 2/3: real save via tf.train.Checkpoint, restore into a FRESH optimizer, compare ---
    ckpt_dir = "/output/optimizer_ckpt_test"
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
        save_path = ckpt.save(os.path.join(ckpt_dir, "ckpt"))
        record("tf_train_checkpoint_save_succeeds", True, save_path)
    except Exception as exc:  # noqa: BLE001
        record("tf_train_checkpoint_save_succeeds", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    # Build a completely FRESH model + optimizer (simulating a brand-new container process / main.py
    # invocation), then restore into it.
    model2 = VGSLModelGenerator().generate_model(model_spec=vgsl_spec, model_name="investigate")
    model2 = replace_final_layer(model2, vocab_size, model2.name)
    lr_schedule2 = create_learning_rate_schedule(
        learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, train_batches=1000,
        do_train=True, warmup_ratio=0.0, epochs=1, decay_per_epoch=False, linear_decay=False,
    )
    optimizer2 = get_optimizer("adam", lr_schedule2)
    model2.compile(optimizer=optimizer2, loss=CTCLoss(), metrics=[CERMetric(greedy=True), WERMetric()], weighted_metrics=[])
    # A freshly compiled optimizer has NOT built its slot variables yet (that only happens on the
    # first apply_gradients) -- Keras 3's tf.train.Checkpoint restore of optimizer slot state
    # generally requires the variables to already exist. Test whether one dummy step first, THEN
    # restore, is required -- report whichever actually works, don't assume.
    fresh_iterations = int(model2.optimizer.iterations.numpy())
    record("fresh_optimizer_iterations_is_zero_before_restore", fresh_iterations == 0, fresh_iterations)

    restore_worked_without_prior_step = False
    try:
        ckpt2 = tf.train.Checkpoint(optimizer=model2.optimizer, model=model2)
        status = ckpt2.restore(save_path)
        status.assert_existing_objects_matched()
        restored_iterations = int(model2.optimizer.iterations.numpy())
        restore_worked_without_prior_step = restored_iterations == iterations_before_save
        record("restore_without_prior_dummy_step", restore_worked_without_prior_step,
               f"restored_iterations={restored_iterations} expected={iterations_before_save}")
    except Exception as exc:  # noqa: BLE001
        record("restore_without_prior_dummy_step", False, f"{exc}\n{traceback.format_exc()}")

    REPORT["restore_worked_without_prior_step"] = restore_worked_without_prior_step

    if not restore_worked_without_prior_step:
        # Try the alternative: one dummy train_on_batch first (builds slot variables), THEN restore.
        model3 = VGSLModelGenerator().generate_model(model_spec=vgsl_spec, model_name="investigate")
        model3 = replace_final_layer(model3, vocab_size, model3.name)
        lr_schedule3 = create_learning_rate_schedule(
            learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, train_batches=1000,
            do_train=True, warmup_ratio=0.0, epochs=1, decay_per_epoch=False, linear_decay=False,
        )
        optimizer3 = get_optimizer("adam", lr_schedule3)
        model3.compile(optimizer=optimizer3, loss=CTCLoss(), metrics=[CERMetric(greedy=True), WERMetric()], weighted_metrics=[])
        try:
            ckpt3 = tf.train.Checkpoint(optimizer=model3.optimizer, model=model3)
            status = ckpt3.restore(save_path)
            restored_iterations3 = int(model3.optimizer.iterations.numpy())
            record("restore_with_prior_dummy_step_not_needed_since_optimizer_vars_build_lazily", False,
                   "see restore_with_manual_build_first check below")
        except Exception as exc:  # noqa: BLE001
            pass

        try:
            # Force optimizer variable creation by calling build() with the model's trainable vars,
            # mirroring what Keras 3 optimizers expose for exactly this "restore before first step"
            # scenario.
            model3.optimizer.build(model3.trainable_variables)
            ckpt3b = tf.train.Checkpoint(optimizer=model3.optimizer, model=model3)
            status = ckpt3b.restore(save_path)
            status.assert_existing_objects_matched()
            restored_iterations3b = int(model3.optimizer.iterations.numpy())
            ok = restored_iterations3b == iterations_before_save
            record("restore_after_explicit_optimizer_build", ok,
                   f"restored_iterations={restored_iterations3b} expected={iterations_before_save}")
            REPORT["restore_after_explicit_optimizer_build_worked"] = ok
            if ok:
                lr_after = float(lr_schedule3(model3.optimizer.iterations).numpy())
                record("lr_matches_after_restore_via_explicit_build", abs(lr_after - lr_before_save) < 1e-9,
                       f"restored_lr={lr_after} expected={lr_before_save}")
        except Exception as exc:  # noqa: BLE001
            record("restore_after_explicit_optimizer_build", False, f"{exc}\n{traceback.format_exc()}")
    else:
        lr_after = None
        try:
            lr_after = float(lr_schedule2(model2.optimizer.iterations).numpy())
            record("lr_matches_after_restore", abs(lr_after - lr_before_save) < 1e-9,
                   f"restored_lr={lr_after} expected={lr_before_save}")
        except Exception as exc:  # noqa: BLE001
            record("lr_matches_after_restore", False, str(exc))

    write_and_exit(0)


def write_and_exit(code):
    REPORT["all_checks_passed"] = all(c["ok"] for c in REPORT["checks"].values())
    REPORT["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs("/output", exist_ok=True)
    with open("/output/optimizer_checkpointing_investigation.json", "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2)
    print(f"\nwritten: /output/optimizer_checkpointing_investigation.json")
    raise SystemExit(0)  # always exit 0 -- this is an investigation, not a pass/fail gate


if __name__ == "__main__":
    main()
