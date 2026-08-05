# Imports

# > Standard library
import os

# > Third-party dependencies
import matplotlib.pyplot as plt
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from model.custom_callback import LoghiCustomCallback


class OptimizerTraceCallback(tf.keras.callbacks.Callback):
    """Experiment 1 instrumentation, not part of the pinned loghi-htr commit.

    Loghi's own `CSVLogger`/`LoghiCustomCallback` only log epoch-level metrics -- there is nothing in
    the pinned commit that records `optimizer.iterations`, the resolved learning rate, or running
    train loss/CER/WER at batch granularity. This callback is the only way to empirically prove (not
    assume) that the optimizer and LR schedule stay continuous across the whole single-container
    epoch, which is the entire point of Experiment 1 -- and, since this run is deliberately one single
    ~16h Keras epoch (no shard boundaries to log at), it is also the *only* source of any training
    progress signal before the run finishes. Keras already computes running train loss/CER/WER and
    passes them into every batch-end callback via `logs` -- previously ignored here, now captured for
    free rather than re-derived. Every failure mode here is swallowed and logged to stderr rather than
    raised -- this callback must never be able to crash a multi-hour training run for a purely
    observational side effect.
    """

    def __init__(self, output_path: str, log_every_n_steps: int = 500) -> None:
        super().__init__()
        self._output_path = output_path
        self._log_every_n_steps = max(1, int(log_every_n_steps))
        self._file = None
        self._next_log_threshold = self._log_every_n_steps
        """Threshold-based, not `iterations % N == 0` -- an exact-multiple check silently produces no
        log rows at all whenever Keras advances `optimizer.iterations` by more than 1 between
        `on_train_batch_end` calls (observed for real in a dry run: batched execution and/or
        mixed-precision loss-scale calibration skipping early `apply_gradients` calls means the exact
        multiple can be stepped over entirely). `iterations >= threshold` can never be skipped this way
        -- it fires on the first call where it is true, then the threshold advances past it."""

    def _resolve_iterations(self) -> int | None:
        try:
            return int(self.model.optimizer.iterations.numpy())
        except Exception as exc:  # noqa: BLE001 -- observational only, must never raise
            print(f"[optimizer_trace] failed to read optimizer.iterations: {exc}")
            return None

    def _resolve_learning_rate(self, iterations: int | None):
        try:
            lr = self.model.optimizer.learning_rate
            if callable(lr):
                step = iterations if iterations is not None else self.model.optimizer.iterations
                return float(lr(step).numpy())
            return float(tf.keras.backend.get_value(lr))
        except Exception as exc:  # noqa: BLE001 -- observational only, must never raise
            print(f"[optimizer_trace] failed to read learning_rate: {exc}")
            return None

    def _log_row(self, *, tag: str, logs: dict | None) -> None:
        if self._file is None:
            return
        try:
            iterations = self._resolve_iterations()
            lr_value = self._resolve_learning_rate(iterations)
            logs = logs or {}
            loss = logs.get("loss")
            cer = logs.get("CER_metric")
            wer = logs.get("WER_metric")

            def _cell(v):
                return "" if v is None else v

            self._file.write(
                f"{tag},{_cell(iterations)},{_cell(lr_value)},{_cell(loss)},{_cell(cer)},{_cell(wer)}\n"
            )
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 -- observational only, must never raise
            print(f"[optimizer_trace] failed to write row: {exc}")

    def on_train_begin(self, logs=None):
        try:
            self._file = open(self._output_path, "w", encoding="utf-8")
            self._file.write("tag,optimizer_iterations,learning_rate,loss,CER_metric,WER_metric\n")
            self._file.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[optimizer_trace] failed to open {self._output_path}: {exc}")
            self._file = None
        self._log_row(tag="train_begin", logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        iterations = self._resolve_iterations()
        if iterations is not None and iterations >= self._next_log_threshold:
            self._log_row(tag=f"batch_end_step_{iterations}", logs=logs)
            while self._next_log_threshold <= iterations:
                self._next_log_threshold += self._log_every_n_steps

    def on_train_end(self, logs=None):
        self._log_row(tag="train_end", logs=logs)
        if self._file is not None:
            try:
                self._file.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[optimizer_trace] failed to close file: {exc}")


class OptimizerStateCheckpointCallback(tf.keras.callbacks.Callback):
    """Epoch-2 continuation instrumentation, not part of the pinned loghi-htr commit.

    Saves the FULL optimizer state (Adam slot variables, the mixed_float16 LossScaleOptimizer
    wrapper's own state, and optimizer.iterations) via `tf.train.Checkpoint` at the end of training --
    the one thing Loghi's own checkpoint format (`custom_callback.py::_save_model`'s `clone_model()` +
    `.save()`) has never captured, confirmed repeatedly in this project's own investigation.

    Verified empirically, in the real pinned container, BEFORE this callback was ever used for a real
    run (`training/experiment_2_overrides/investigate_optimizer_checkpointing.py`):
    `tf.train.Checkpoint(optimizer=model.optimizer, model=model)` correctly round-trips this state,
    including under mixed_float16's automatic LossScaleOptimizer wrapping, with no dummy training step
    needed on the fresh optimizer before restoring, and an exact learning-rate match after restore.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self._output_dir = output_dir

    def on_train_end(self, logs=None):
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
            save_path = ckpt.save(os.path.join(self._output_dir, "ckpt"))
            iterations = int(self.model.optimizer.iterations.numpy())
            print(f"[optimizer_state_checkpoint] saved to {save_path} at optimizer.iterations={iterations}")
        except Exception as exc:  # noqa: BLE001 -- observational/best-effort, must never crash training
            print(f"[optimizer_state_checkpoint] FAILED to save: {exc}")


def train_model(model: tf.keras.Model,
                config: Config,
                training_dataset: tf.data.Dataset,
                validation_dataset: tf.data.Dataset,
                data_manager: DataManager) -> tf.keras.callbacks.History:
    """
    Trains a Keras model using the provided training and validation datasets,
    along with additional arguments.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    config : Config
        A Config object containing model and training configurations.
    training_dataset : tf.data.Dataset
        The dataset to be used for training.
    validation_dataset : tf.data.Dataset
        The dataset to be used for validation.
    data_manager : DataManager
        A DataManager containing additional information like character list.

    Returns
    -------
    tf.keras.callbacks.History
        The training history object containing information about the training
        process (e.g., loss values, metrics).

    Notes
    -----
    This function sets up a custom training routine for a Keras model, with
    logging and early stopping functionalities. The actual training process
    depends on the specific model and data provided.
    """

    # CSV logger
    log_filename = os.path.join(config["output"], 'log.csv')
    logging_callback = tf.keras.callbacks.CSVLogger(
        log_filename, separator=",", append=True)

    # Loghi custom callback
    loghi_custom_callback = \
        LoghiCustomCallback(save_best=True,
                            save_checkpoint=config["output_checkpoints"],
                            output=config["output"],
                            tokenizer=data_manager.tokenizer,
                            config=config,
                            normalization_file=config["normalization_file"])

    # Experiment 1 instrumentation: proves optimizer/LR continuity across the whole epoch (see
    # OptimizerTraceCallback's own docstring above -- not part of the pinned loghi-htr commit).
    optimizer_trace_callback = OptimizerTraceCallback(
        output_path=os.path.join(config["output"], 'optimizer_trace.csv'),
        log_every_n_steps=500,
    )

    # Epoch-2 continuation instrumentation: persists full optimizer state so the NEXT container
    # invocation can genuinely resume it (see OptimizerStateCheckpointCallback's own docstring above).
    optimizer_state_callback = OptimizerStateCheckpointCallback(
        output_dir=os.path.join(config["output"], 'optimizer_state'),
    )

    # Add all default callbacks
    callbacks = [logging_callback, loghi_custom_callback, optimizer_trace_callback, optimizer_state_callback]

    # If we defined an early stopping patience, add it to the callbacks
    if config["early_stopping_patience"] > 0 and validation_dataset:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_CER_metric',
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)

    # Determine the number of steps per epoch
    # NOTE: None means that the number of steps is equal to the number of
    # batches in the dataset (default behavior)
    # FIXME: steps_per_epoch is not working properly
    steps_per_epoch = config["steps_per_epoch"] \
        if config["steps_per_epoch"] else None

    # Train the model
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=config["epochs"],
        callbacks=callbacks,
        shuffle=True,
        steps_per_epoch=steps_per_epoch,
        verbose=config["training_verbosity_mode"]
    )

    return history


def plot_metric(metric, history, title, output_path, plot_validation_metric):
    plt.style.use("ggplot")
    plt.figure()

    # Check if the metric exists in the history
    if metric not in history.history:
        raise ValueError(f"Metric '{metric}' not found in history")

    # Plot the training metric
    plt.plot(history.history[metric], label='Training ' + metric)

    # Plot the validation metric if requested
    if plot_validation_metric:
        print(f"Validation metric: val_{metric}")
        val_metric = f"val_{metric}"
        if val_metric not in history.history:
            raise ValueError(f"Validation metric '{val_metric}' not found in history")
        plt.plot(history.history[val_metric], label=f"Validation {metric}")

    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(metric)
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    plt.close()


def plot_training_history(history: tf.keras.callbacks.History,
                          output_path: str,
                          plot_validation: bool = False) -> None:
    """
    Plots the training history of a Keras model, including loss and Character
    Error Rate (CER).

    Parameters
    ----------
    history : tf.keras.callbacks.History
        The training history object returned by a model training process,
        containing metrics like loss and CER over epochs.
    output_path : str
        Path to save the plots.
    plot_validation : bool, default False
        Whether to plot validation metrics.

    Notes
    -----
    This function generates and saves two plots: one for training loss and the
    other for Character Error Rate (CER).
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plot_metric(metric="loss",
                history=history,
                title="Training Loss",
                output_path=os.path.join(output_path, 'loss_plot.png'),
                plot_validation_metric=plot_validation)
    plot_metric(metric="CER_metric",
                history=history,
                title="Character Error Rate (CER)",
                output_path=os.path.join(output_path, 'cer_plot.png'),
                plot_validation_metric=plot_validation)
