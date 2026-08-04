"""Phase 4 + Phase 6 (partial): architecture report and bounded in-container smoke test for
Experiment 2's from-scratch VGSL model. Runs inside the pinned loghi-htr container -- vgslify and
TensorFlow are only available there, not on the host. Writes a JSON report to /output.

Never touches the real corpus or a real optimizer step over real data -- all inputs here are small
synthetic tensors, matching the brief's "do not perform optimizer steps against the complete real
corpus during preflight."
"""

import json
import os
import time
import traceback

os.environ["TF_USE_LEGACY_KERAS"] = "0"

import numpy as np
import tensorflow as tf
from vgslify.generator import VGSLModelGenerator

from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.optimization import LoghiLearningRateSchedule, create_learning_rate_schedule, get_optimizer
from model.management import get_model_library
from model.replacing import replace_final_layer

REPORT = {"phase": "experiment_2_architecture_and_preflight", "checks": {}}


def record(name, ok, detail=None):
    REPORT["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail if detail is not None else ''}")


def main():
    # --- Resolve the exact architecture spec Loghi's own build_predefined_model uses ---
    library = get_model_library()
    if "recommended" not in library:
        record("recommended_key_resolves", False, "no 'recommended' key in get_model_library()")
        write_and_exit(1)
        return
    vgsl_spec = library["recommended"]
    record("recommended_key_resolves", True, vgsl_spec)
    REPORT["vgsl_specification"] = vgsl_spec

    # --- Build from random initialization ---
    try:
        generator = VGSLModelGenerator()
        model = generator.generate_model(model_spec=vgsl_spec, model_name="experiment_2_recommended_scratch")
        record("architecture_parses_and_instantiates", True, f"model name={model.name}")
    except Exception as exc:  # noqa: BLE001
        record("architecture_parses_and_instantiates", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    # --- Layer sequence + parameter counts, BEFORE final-layer resize ---
    def describe_layers(m):
        layers = []
        for layer in m.layers:
            try:
                out_shape = str(layer.output.shape)
            except Exception:  # noqa: BLE001
                out_shape = None
            layers.append({
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": out_shape,
                "param_count": int(layer.count_params()),
            })
        return layers

    REPORT["layers_before_vocab_resize"] = describe_layers(model)
    REPORT["params_before_vocab_resize"] = {
        "total": int(model.count_params()),
        "trainable": int(sum(int(tf.size(w)) for w in model.trainable_weights)),
        "non_trainable": int(sum(int(tf.size(w)) for w in model.non_trainable_weights)),
    }

    # --- Resize final layer to the REAL vocabulary size (Loghi always does this for a from-scratch
    # build, regardless of the literal Fs92 in the VGSL string -- management.py::customize_model) ---
    charlist_path = "/charlist_inventory/experiment_2_character_inventory.json"
    try:
        inventory = json.load(open(charlist_path, encoding="utf-8"))
        chars = inventory["scratch_model_vocabulary"]["characters"]
        from utils.text import Tokenizer
        tokenizer = Tokenizer(chars)
        vocab_size = len(tokenizer)
        record("character_list_valid_and_tokenizer_builds", True,
               f"{len(chars)} raw chars -> tokenizer vocabulary size {vocab_size}")
    except Exception as exc:  # noqa: BLE001
        record("character_list_valid_and_tokenizer_builds", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    try:
        model = replace_final_layer(model, vocab_size, model.name)
        record("final_layer_resize_to_real_vocabulary", True, f"vocab_size={vocab_size}")
    except Exception as exc:  # noqa: BLE001
        record("final_layer_resize_to_real_vocabulary", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    REPORT["layers_after_vocab_resize"] = describe_layers(model)
    REPORT["params_after_vocab_resize"] = {
        "total": int(model.count_params()),
        "trainable": int(sum(int(tf.size(w)) for w in model.trainable_weights)),
        "non_trainable": int(sum(int(tf.size(w)) for w in model.non_trainable_weights)),
    }
    REPORT["vocabulary_size_used"] = vocab_size

    # --- Compile with the real loss/metrics/optimizer/LR schedule Experiment 2 will actually use ---
    try:
        lr_schedule = create_learning_rate_schedule(
            learning_rate=0.0001, decay_rate=0.99, decay_steps=-1,
            train_batches=35133,  # real value: ceil(562123/16), for a representative schedule shape
            do_train=True, warmup_ratio=0.0, epochs=1, decay_per_epoch=False, linear_decay=False,
        )
        optimizer = get_optimizer("adam", lr_schedule)
        model.compile(optimizer=optimizer, loss=CTCLoss(),
                      metrics=[CERMetric(greedy=True), WERMetric()], weighted_metrics=[])
        record("compiles_with_real_loss_optimizer_lr_schedule", True,
               f"optimizer=adam lr=0.0001 decay_rate=0.99 decay_steps=train_batches")
    except Exception as exc:  # noqa: BLE001
        record("compiles_with_real_loss_optimizer_lr_schedule", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    # --- Bounded forward pass: small synthetic batch. VGSL "None,None,64,1" means
    # (batch, width[variable], height=64[fixed], channels) -- confirmed via model.input_shape,
    # NOT (batch, height, width, channels) as first assumed (that assumption failed for real: Keras
    # rejected a (2, 64, 256, 1) array against expected (None, None, 64, 1) because 64 landed in the
    # variable width slot instead of the fixed height slot).
    batch_size, width, height, channels = 2, 256, 64, 1
    x = np.random.rand(batch_size, width, height, channels).astype("float32")
    try:
        y_pred = model(x, training=False)
        record("bounded_forward_pass_succeeds", True, f"output shape={tuple(y_pred.shape)}")
    except Exception as exc:  # noqa: BLE001
        record("bounded_forward_pass_succeeds", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    time_steps = int(y_pred.shape[1])
    max_label_len = min(20, max(1, time_steps - 1))
    y_true = np.random.randint(1, vocab_size, size=(batch_size, max_label_len)).astype("int64")

    # --- Bounded backward pass: one real train_on_batch call with the real CTCLoss ---
    try:
        loss_value = model.train_on_batch(x, y_true)
        record("bounded_backward_pass_succeeds", True, f"train_on_batch loss={loss_value}")
    except Exception as exc:  # noqa: BLE001
        record("bounded_backward_pass_succeeds", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    try:
        iters_after_one_step = int(model.optimizer.iterations.numpy())
        record("optimizer_iterations_advance_on_real_step", iters_after_one_step >= 1,
               f"optimizer.iterations={iters_after_one_step} after 1 train_on_batch call")
    except Exception as exc:  # noqa: BLE001
        record("optimizer_iterations_advance_on_real_step", False, str(exc))

    # --- Checkpoint save + reload, matching LoghiCustomCallback._save_model's exact mechanism ---
    ckpt_dir = "/output/preflight_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, "model.keras")
    try:
        with tf.device("/CPU:0"):
            unfrozen = tf.keras.models.clone_model(model)
            unfrozen.set_weights(model.get_weights())
            for layer in unfrozen.layers:
                layer.trainable = True
            unfrozen.save(model_path)
        record("checkpoint_save_succeeds", True, model_path)
    except Exception as exc:  # noqa: BLE001
        record("checkpoint_save_succeeds", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    try:
        reloaded = tf.keras.models.load_model(
            model_path,
            custom_objects={"CERMetric": CERMetric, "WERMetric": WERMetric,
                            "CTCLoss": CTCLoss, "ResidualBlock": ResidualBlock,
                            "LoghiLearningRateSchedule": LoghiLearningRateSchedule},
            compile=False,
        )
        original_weights = model.get_weights()
        reloaded_weights = reloaded.get_weights()
        weights_match = len(original_weights) == len(reloaded_weights) and all(
            np.array_equal(a, b) for a, b in zip(original_weights, reloaded_weights)
        )
        record("checkpoint_reload_succeeds_and_weights_match", weights_match,
               f"{len(reloaded_weights)} weight arrays compared")
    except Exception as exc:  # noqa: BLE001
        record("checkpoint_reload_succeeds_and_weights_match", False, f"{exc}\n{traceback.format_exc()}")
        write_and_exit(1)
        return

    # --- Peak VRAM during this bounded test ---
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        REPORT["peak_vram_bytes_during_smoke_test"] = info["peak"]
        REPORT["peak_vram_mb_during_smoke_test"] = round(info["peak"] / (1024 * 1024), 1)
    except Exception as exc:  # noqa: BLE001
        REPORT["peak_vram_bytes_during_smoke_test"] = None
        REPORT["vram_check_error"] = str(exc)

    write_and_exit(0)


def write_and_exit(code):
    REPORT["all_checks_passed"] = all(c["ok"] for c in REPORT["checks"].values())
    REPORT["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs("/output", exist_ok=True)
    with open("/output/architecture_report_and_preflight.json", "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2)
    print(f"\nwritten: /output/architecture_report_and_preflight.json")
    print(f"all_checks_passed={REPORT['all_checks_passed']}")
    raise SystemExit(0 if REPORT["all_checks_passed"] else code)


if __name__ == "__main__":
    main()
