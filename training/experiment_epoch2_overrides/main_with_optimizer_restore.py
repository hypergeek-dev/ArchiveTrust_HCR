# Imports
import argparse
# > Standard library
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import time
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
# Data handling
from data.data_handling import initialize_data_manager

# Model-specific
from data.augmentation import make_augment_model, visualize_augments
from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.management import load_or_create_model, customize_model
from model.optimization import create_learning_rate_schedule, get_optimizer, \
    LoghiLearningRateSchedule
from modes.training import train_model, plot_training_history
from modes.evaluation import perform_evaluation

# Setup and configuration
from setup.arg_parser import get_args
from setup.config import Config
from setup.environment import setup_environment, setup_logging

# Utilities
from utils.print import summarize_model
from utils.text import Tokenizer


def _dummy_step_to_build_optimizer_variables(model) -> None:
    """Forces the optimizer to create its slot variables (Adam m/v moments, the LossScaleOptimizer
    wrapper's own step_counter, etc.) through the EXACT same code path real training uses --
    `apply_gradients` via one real `train_on_batch` call -- rather than an explicit `optimizer.
    build(...)` call. Empirically, explicit `.build()` produced a real dtype mismatch on restore
    (`step_counter` created as float by `.build()` vs. int64 in a checkpoint from real training) that
    a dummy training step does not: same code path in, same variable structure out. The dummy step's
    effect on weights and optimizer state is immaterial -- both get overwritten by the restore that
    follows immediately after."""
    import numpy as np

    channels = model.input_shape[-1]
    height = model.input_shape[2]
    dummy_x = np.random.rand(1, 128, height, channels).astype("float32")
    # CTCLoss (model/losses.py::call) does `y_true = where(y_true > 0, y_true - 1, y_true)` then
    # `label_length = count_nonzero(y_true)` -- a label value of 1 maps to 0 after the shift, which
    # reads as padding/empty, not a real character. A real bug this dummy step caught: labels of all
    # 1s produced "Labels length is zero in batch 0". Using 2/3 (which shift to 1/2, both nonzero)
    # gives CTCLoss a real, valid label sequence to compute a loss against.
    dummy_y = np.array([[2, 3, 2, 3]], dtype="int64")
    model.train_on_batch(dummy_x, dummy_y)


def _restore_optimizer_state_if_present(*, model, optimizer, model_dir: str) -> None:
    """Epoch-2 continuation: NOT part of the pinned loghi-htr commit. Additive and defensive -- if no
    `optimizer_state/` sidecar exists at the mounted `--model` directory (the pristine pretrained
    checkpoint, or any epoch-1 checkpoint that predates this mechanism), this is a silent no-op and
    stock behavior is unchanged: a fresh optimizer/LR-schedule, exactly as before.

    Verified empirically, in the real pinned container, BEFORE this was ever used for a real run
    (`training/experiment_2_overrides/investigate_optimizer_checkpointing.py`):
    `tf.train.Checkpoint(optimizer=optimizer, model=model)` correctly restores Adam's slot variables,
    the mixed_float16 LossScaleOptimizer wrapper's own state, and `optimizer.iterations`, with no prior
    dummy training step needed and an exact learning-rate match immediately after restore.
    """
    optimizer_state_dir = os.path.join(model_dir, "optimizer_state")
    if not os.path.isdir(optimizer_state_dir):
        logging.info("No optimizer_state/ sidecar at %s -- starting with a fresh optimizer (this is "
                    "expected for the pristine pretrained checkpoint or any epoch 1 run).",
                    model_dir)
        return

    latest_ckpt = tf.train.latest_checkpoint(optimizer_state_dir)
    if not latest_ckpt:
        logging.warning("optimizer_state/ directory present at %s but no checkpoint found inside -- "
                        "starting with a fresh optimizer.", optimizer_state_dir)
        return

    # `model` here was just loaded fresh via tf.keras.models.load_model() (load_or_create_model's
    # "load from directory" branch) -- a DIFFERENT Python object than whatever model instance was
    # actually training when the optimizer checkpoint was saved, even though architecturally
    # identical. A freshly-compiled optimizer has not called apply_gradients yet, so its Adam slot
    # variables (m/v moments) do not exist yet. Real, empirical findings from this project's own
    # cross-process tests (two genuinely separate container invocations, not a single-process
    # investigation script):
    #   1. Restoring into such an optimizer with NO prior variable-build step silently produced
    #      optimizer.iterations=0 instead of the true saved value -- wrong, with no error at all.
    #   2. Forcing variable creation via an explicit `optimizer.build(model.trainable_variables)`
    #      call instead raised a real, loud dtype error: `step_counter` was created as float by
    #      `.build()`, but the checkpoint (from real training) has it as int64 -- also wrong, but at
    #      least caught rather than silent.
    #   3. Forcing variable creation via one real dummy `train_on_batch` call -- the same code path
    #      real training itself uses to create these variables -- fixed both failure modes above, but
    #      a THIRD, separate bug remained even then: checkpointing the caller's plain `optimizer`
    #      argument (the un-wrapped Adam instance originally passed into `model.compile(...)`)
    #      instead of `model.optimizer` (what Keras actually attaches to the model after compile,
    #      under the real mixed_float16 policy: a `LossScaleOptimizer` WRAPPING that Adam instance --
    #      a structurally different object with its own `step_counter` and `inner_optimizer`
    #      sub-tree). The plain `optimizer` variable was never touched by the dummy step (`model.
    #      train_on_batch` uses `model.optimizer`, not the caller's local reference) and has none of
    #      that wrapper structure, which is exactly what produced the dtype/shape mismatches above --
    #      not a Keras bug, a bug in which object this function checkpointed. `model.optimizer` is the
    #      one both the save side (`OptimizerStateCheckpointCallback`, via `self.model.optimizer`) and
    #      real training (`model.fit`) actually use.
    _dummy_step_to_build_optimizer_variables(model)

    restore_ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    status = restore_ckpt.restore(latest_ckpt)
    # Deliberately strict, not .expect_partial() -- a silent partial restore is exactly the failure
    # mode that produced optimizer.iterations=0 with no error the first time this was tried for real.
    status.assert_existing_objects_matched()
    iterations = int(model.optimizer.iterations.numpy())
    logging.info("Restored optimizer state from %s -- optimizer.iterations=%d", latest_ckpt, iterations)
    print(f"[optimizer_state_restore] restored from {latest_ckpt}, optimizer.iterations={iterations}")


def main(args=None):
    """ Main function for the program """
    setup_logging()

    # Get the arguments
    if args is None:
        parsed_args = get_args()
        print('parsed_args: ', parsed_args)
    else:
        parsed_args = get_args(args)
        # parsed_args = args
    config = Config(*parsed_args)

    # Set up the environment
    strategy = setup_environment(config)

    # Create the output directory if it doesn't exist
    if config["output"]:
        os.makedirs(config["output"], exist_ok=True)

    # Determine the path to the tokenizer file
    json_path = None

    if config["tokenizer"]:
        json_path = config["tokenizer"]
    elif os.path.isdir(config["model"]):
        json_path = next(
            (os.path.join(config["model"], fname) for fname in ["tokenizer.json", "charlist.txt"]
             if os.path.exists(os.path.join(config["model"], fname))),
            None
        )

    # Load the tokenizer if a valid path was found
    if json_path and not config["replace_final_layer"]:
        tokenizer = Tokenizer.load_from_file(json_path)
    else:
        tokenizer = None  # Indicate that a new tokenizer will be created later

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(config, custom_objects)
        augmentation_model = make_augment_model(config, model.input_shape[-1])

        if config["visualize_augments"] and augmentation_model:
            visualize_augments(augmentation_model,
                               config["output"],
                               model.input_shape[-1])

        # Initialize the DataManager
        data_manager = initialize_data_manager(config, tokenizer, model,
                                               augmentation_model)

        # Replace the tokenizer with the one from the data manager
        tokenizer = data_manager.tokenizer
        logging.info("Tokenizer size: %s tokens", len(tokenizer))

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, config, tokenizer)

        # Save the tokenizer
        tokenizer.save_to_json(os.path.join(config["output"],
                                            "tokenizer.json"))

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            learning_rate=config["learning_rate"],
            decay_rate=config["decay_rate"],
            decay_steps=config["decay_steps"],
            train_batches=data_manager.get_train_batches(),
            do_train=config["train_list"],
            warmup_ratio=config["warmup_ratio"],
            epochs=config["epochs"],
            decay_per_epoch=config["decay_per_epoch"],
            linear_decay=config["linear_decay"])

        # Create the optimizer
        optimizer = get_optimizer(config["optimizer"], lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=CTCLoss(),
                      metrics=[CERMetric(greedy=config["greedy"],
                                         beam_width=config["beam_width"]),
                               WERMetric()],
                      weighted_metrics=[])

        # Epoch-2 continuation: restore full optimizer state if a sidecar checkpoint is present.
        # NOT part of the pinned loghi-htr commit -- see _restore_optimizer_state_if_present's own
        # docstring. A silent no-op for every stock invocation (pristine checkpoint, or any run that
        # predates this mechanism).
        if os.path.isdir(config["model"]):
            _restore_optimizer_state_if_present(model=model, optimizer=optimizer, model_dir=config["model"])

    # Print the model summary
    logging.info("Model Summary:")
    model.summary(line_length=100)

    # Store the model info (i.e., git hash, args, model summary, etc.)
    config.update_config_key("model", summarize_model(model))
    config.update_config_key("model_name", model.name)
    config.update_config_key("model_channels", model.input_shape[-1])
    config.save()

    # Store timestamps
    timestamps = {'start_time': time.time()}

    # Train the model
    if config["train_list"]:
        tick = time.time()

        history = train_model(model,
                              config,
                              data_manager.datasets["train"],
                              data_manager.datasets["evaluation"],
                              data_manager)
        # Plot the training history
        plot_training_history(history=history,
                              output_path=config["output"],
                              plot_validation=bool(config["validation_list"]))

        timestamps['Training'] = time.time() - tick

    # Evaluation modes and their corresponding conditions
    evaluation_modes = [
        ("validation", config["do_validate"],
         "Validation results are without special markdown tags"),
        ("test", config["test_list"],
         "Test results are without special markdown tags"),
        ("inference", config["inference_list"], None)
    ]

    for mode, condition, warning in evaluation_modes:
        if condition:
            if warning:
                logging.warning(warning)
            tick = time.time()
            perform_evaluation(config, model, data_manager, mode)
            timestamps[mode.capitalize()] = time.time() - tick

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    for key, value in list(timestamps.items())[1:]:
        logging.info("%s completed in %.2f seconds", key, value)
    logging.info("Total time: %.2f seconds",
                 time.time() - timestamps['start_time'])


if __name__ == "__main__":
    main()
