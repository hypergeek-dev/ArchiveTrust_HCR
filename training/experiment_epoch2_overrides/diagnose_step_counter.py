"""Targeted diagnostic: what exactly is "step_counter" in the saved checkpoint, and what does the
freshly-loaded model's optimizer create it as, before any restore is attempted? Read-only inspection,
no restore attempt.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import json

import tensorflow as tf

from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.management import load_or_create_model
from model.metrics import CERMetric, WERMetric
from model.optimization import LoghiLearningRateSchedule, create_learning_rate_schedule, get_optimizer
from setup.config import Config

if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_float16"))

CKPT_DIR = "/model/optimizer_state"
latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR)
print("latest_ckpt:", latest_ckpt)

print("\n=== variables in the checkpoint mentioning step_counter or iterations ===")
for name, shape in tf.train.list_variables(latest_ckpt):
    if "step_counter" in name.lower() or "iteration" in name.lower():
        print(f"  {name}  shape={shape}")
        try:
            dtype_reader = tf.train.load_checkpoint(latest_ckpt)
            dt = dtype_reader.get_variable_to_dtype_map().get(name)
            print(f"    dtype in checkpoint: {dt}")
        except Exception as exc:  # noqa: BLE001
            print(f"    (dtype read failed: {exc})")

# --- Now build the SAME way main.py does: load model from /model, compile, inspect optimizer ---
custom_objects = {"CERMetric": CERMetric, "WERMetric": WERMetric, "CTCLoss": CTCLoss,
                  "ResidualBlock": ResidualBlock, "LoghiLearningRateSchedule": LoghiLearningRateSchedule}

class FakeConfig(dict):
    def __getitem__(self, key):
        return super().get(key)

config = FakeConfig({
    "model": "/model", "output": "/output_diag", "tokenizer": None, "replace_final_layer": False,
    "replace_recurrent_layer": None, "freeze_conv_layers": False, "freeze_recurrent_layers": False,
    "freeze_dense_layers": False, "use_float32": False, "gpu": "0", "model_name": None,
})
os.makedirs("/output_diag", exist_ok=True)

from model.management import load_model_from_directory
model = load_model_from_directory("/model", output_directory="/output_diag", custom_objects=custom_objects)
print("\nmodel loaded. model.optimizer before explicit compile:", model.optimizer)

lr_schedule = create_learning_rate_schedule(
    learning_rate=0.0001, decay_rate=0.99, decay_steps=-1, train_batches=1000,
    do_train=True, warmup_ratio=0.0, epochs=1, decay_per_epoch=False, linear_decay=False,
)
optimizer = get_optimizer("adam", lr_schedule)
model.compile(optimizer=optimizer, loss=CTCLoss(), metrics=[CERMetric(greedy=True), WERMetric()], weighted_metrics=[])
print("model.optimizer after explicit compile:", type(model.optimizer).__name__)

print("\n=== optimizer variables mentioning step_counter or iterations, BEFORE any restore/dummy step ===")
try:
    for v in model.optimizer.variables:
        if "step_counter" in v.name.lower() or "iteration" in v.name.lower():
            print(f"  name={v.name} dtype={v.dtype} shape={v.shape}")
except Exception as exc:  # noqa: BLE001
    print(f"(could not enumerate optimizer.variables before build: {exc})")

# force build via a dummy step, then inspect again
import numpy as np
channels = model.input_shape[-1]
height = model.input_shape[2]
dummy_x = np.random.rand(1, 128, height, channels).astype("float32")
dummy_y = np.array([[2, 3, 2, 3]], dtype="int64")
model.train_on_batch(dummy_x, dummy_y)

print("\n=== optimizer variables mentioning step_counter or iterations, AFTER one dummy train_on_batch ===")
for v in model.optimizer.variables:
    if "step_counter" in v.name.lower() or "iteration" in v.name.lower():
        print(f"  name={v.name} dtype={v.dtype} shape={v.shape}")

print("\n=== attempting the actual restore now ===")
try:
    restore_ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    status = restore_ckpt.restore(latest_ckpt)
    status.assert_existing_objects_matched()
    print("RESTORE SUCCEEDED. optimizer.iterations =", int(model.optimizer.iterations.numpy()))
except Exception as exc:  # noqa: BLE001
    print(f"RESTORE FAILED: {exc}")

print("\nDONE")
