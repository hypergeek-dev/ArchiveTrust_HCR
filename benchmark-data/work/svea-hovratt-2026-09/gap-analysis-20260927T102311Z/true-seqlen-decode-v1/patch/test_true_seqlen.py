"""Unit tests A-C for the true-seqlen-decode-v1 fix, run inside the pinned
loghi-htr container (needs its tensorflow/numpy). Test D (regression on the
real previously-confirmed changed lines) is covered separately by the full
batch-invariance matrix re-run against real benchmark crops, not here.

Run: docker run --rm --entrypoint python3 <image> \
       -v <this file>:/test_true_seqlen.py:ro \
       -v decoding.py:/src/loghi-htr/src/utils/decoding.py:ro \
       /test_true_seqlen.py
"""
import sys
sys.path.insert(0, "/src/loghi-htr/src")

import numpy as np  # noqa: E402
from utils.decoding import decode_batch_predictions  # noqa: E402


class FakeTokenizer:
    """Minimal tokenizer: index i -> chr(ord('a')+i-1), 0 reserved for blank/pad."""

    def decode(self, arr):
        chars = []
        for v in arr:
            v = int(v)
            if v <= 0:
                continue
            chars.append(chr(ord("a") + v - 1))
        return "".join(chars)


def true_width_from_images(images_np):
    """Exact reimplementation of the modes_utils.py patch's width detector,
    for testing the detection logic itself against known synthetic inputs."""
    is_batch_pad_col = np.all(images_np == -10.0, axis=(2, 3))
    true_width = np.zeros(images_np.shape[0], dtype=np.int64)
    for b in range(images_np.shape[0]):
        nonpad = np.where(~is_batch_pad_col[b])[0]
        true_width[b] = (int(nonpad.max()) + 1) if nonpad.size else 1
    return true_width


def test_A_width_detection_mixed_batch():
    """A: per-sample true widths differ correctly in a mixed-width batch, and
    do NOT depend on which other samples share the batch."""
    # 3 samples: true widths 10, 40, 25; batched (padded to 40) with -10 fill.
    rng = np.random.default_rng(0)
    batch_a = np.full((3, 40, 4, 1), -10.0, dtype=np.float32)
    batch_a[0, :10] = rng.uniform(-0.5, 0.5, size=(10, 4, 1))
    batch_a[1, :40] = rng.uniform(-0.5, 0.5, size=(40, 4, 1))
    batch_a[2, :25] = rng.uniform(-0.5, 0.5, size=(25, 4, 1))
    widths_a = true_width_from_images(batch_a)
    assert list(widths_a) == [10, 40, 25], widths_a

    # Same sample 0 (width 10), now batched with a much-wider companion (80),
    # so the -10 padding suffix is longer -- true width must still read 10.
    batch_b = np.full((2, 80, 4, 1), -10.0, dtype=np.float32)
    batch_b[0, :10] = batch_a[0, :10]
    batch_b[1, :80] = rng.uniform(-0.5, 0.5, size=(80, 4, 1))
    widths_b = true_width_from_images(batch_b)
    assert widths_b[0] == 10, widths_b
    print("A: PASS -- per-sample true width is correct and companion-independent")


def one_hot_prob_row(vocab, idx, hi=0.96):
    row = np.full(vocab, (1.0 - hi) / (vocab - 1), dtype=np.float32)
    row[idx] = hi
    return row


# tf.nn.ctc_*_decoder treats the LAST class index (vocab - 1) as blank, not
# index 0 -- decode_batch_predictions then shifts the surviving raw symbol
# indices by +1 to land in the tokenizer's own index space (0 = pad/skip).
VOCAB = 5
BLANK = VOCAB - 1


def test_B_padding_timesteps_excluded_from_decode():
    """B: batch-max padding timesteps are excluded from decoding when the true
    sequence_length is supplied, and included (the bug) when it is not.

    `pred` is a probability distribution per timestep (decode_batch_predictions
    takes tf.math.log(pred) directly), not raw logits.
    """
    tok = FakeTokenizer()
    T = 20
    true_ts = 6  # only the first 6 timesteps are real content for this sample

    pred = np.zeros((1, T, VOCAB), dtype=np.float32)
    # Real content (timesteps 0..5): raw symbol 0 repeated (merges to one 'a'),
    # then blank (index BLANK) for the rest of the real content region.
    real_seq = [0, 0, 0, BLANK, BLANK, BLANK]
    for t, idx in enumerate(real_seq):
        pred[0, t] = one_hot_prob_row(VOCAB, idx)
    # "Padding" timesteps (6..19): high-confidence for a *different real*
    # symbol (raw index 2 -> 'c'), simulating a spurious character leaking in
    # from batch-padding logits if a decoder is (wrongly) allowed to read
    # past true_ts. merge_repeated collapses the whole run into one 'c'.
    for t in range(true_ts, T):
        pred[0, t] = one_hot_prob_row(VOCAB, 2)

    _, buggy_text = decode_batch_predictions(pred, tok, greedy=True, beam_width=1)[0]
    _, fixed_text = decode_batch_predictions(
        pred, tok, greedy=True, beam_width=1, sequence_lengths=[true_ts]
    )[0]

    assert fixed_text == "a", fixed_text
    assert buggy_text == "ac", buggy_text
    assert buggy_text != fixed_text, (buggy_text, fixed_text)
    print(f"B: PASS -- buggy(uniform)={buggy_text!r} fixed(true_len)={fixed_text!r}")


def test_C_same_crop_invariant_to_companions():
    """C: the same crop's logits, decoded with its own true sequence_length,
    give the same result whether pred.shape[1] (the batch's own width) is 20
    or 60 -- i.e. once the per-sample true length is passed, batch composition
    no longer changes the decode of that sample."""
    tok = FakeTokenizer()
    true_ts = 6
    # raw symbols 1,1 (merge->1), then BLANK, then 3 (merge->3), then BLANK,BLANK
    real_seq = [1, 1, BLANK, 3, BLANK, BLANK]  # -> raw [1,3] -> +1 -> "bd"

    def make_pred(total_T):
        pred = np.zeros((1, total_T, VOCAB), dtype=np.float32)
        for t, idx in enumerate(real_seq):
            pred[0, t] = one_hot_prob_row(VOCAB, idx)
        for t in range(true_ts, total_T):
            pred[0, t] = one_hot_prob_row(VOCAB, 2)  # spurious padding leak, excluded by true_ts
        return pred

    pred_narrow_batch = make_pred(20)
    pred_wide_batch = make_pred(60)

    _, text_narrow = decode_batch_predictions(
        pred_narrow_batch, tok, greedy=True, beam_width=1, sequence_lengths=[true_ts]
    )[0]
    _, text_wide = decode_batch_predictions(
        pred_wide_batch, tok, greedy=True, beam_width=1, sequence_lengths=[true_ts]
    )[0]

    assert text_narrow == text_wide == "bd", (text_narrow, text_wide)
    print(f"C: PASS -- invariant to batch width: narrow={text_narrow!r} wide={text_wide!r}")


if __name__ == "__main__":
    test_A_width_detection_mixed_batch()
    test_B_padding_timesteps_excluded_from_decode()
    test_C_same_crop_invariant_to_companions()
    print("ALL UNIT TESTS PASSED")
