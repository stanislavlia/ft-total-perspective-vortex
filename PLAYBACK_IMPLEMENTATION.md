# Playback Feature Implementation Plan (Option 2)

Streaming prediction from raw continuous EEG data with a pre-trained model.

## Overview

Train mode saves a fitted pipeline to disk. Predict mode loads it and walks
through the raw EDF file chronologically, extracting and classifying epochs
one by one as events are encountered — simulating real-time BCI inference.

---

## Step 1: Model Serialization in Train Mode

**File:** `src/cli_commands/run.py`

- After training the pipeline, save it to disk using `joblib.dump(pipeline, path)`.
- Train mode should print the path where the model was saved.
- The save path can be a default (e.g., `models/` directory) or user-specified via CLI.

**CLI change in `src/mybci.py`:**
- Add `--model-output` / `-o` option to `run` command (optional, with sensible default).

**Expected behavior:**
```bash
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m train
# → "Model saved to models/S001_motor_imagery_left_right_hand.joblib"
```

---

## Step 2: Add `--model` CLI Option for Predict Mode

**File:** `src/mybci.py`

- Add `--model` option to `run` command.
- Required when `--mode predict`, ignored otherwise.
- Accepts a path to a `.joblib` file.

**Expected usage:**
```bash
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m predict \
  --model models/S001_motor_imagery_left_right_hand.joblib
```

---

## Step 3: Rewrite Predict Mode as Streaming Playback

**File:** `src/cli_commands/run.py`

Replace the current `run_predict_mode` logic. New flow:

1. **Load the saved pipeline** from the `--model` path using `joblib.load()`.
2. **Load raw EDF files** for the given subject/task_type/paradigm via `data_loader`.
   - Need a new method or to reuse existing loading logic that returns raw
     continuous data (not pre-extracted epochs).
3. **Apply bandpass filter** to the raw data (8-30 Hz).
4. **Extract events** from annotations, sorted chronologically.
5. **For each event:**
   - Determine the event label (ground truth).
   - Extract the data window: `raw[channels, event_sample + t_min*sfreq : event_sample + t_max*sfreq]`.
   - Reshape to `(1, n_channels, n_times)`.
   - Start timer.
   - Call `pipeline.predict(epoch_data)`.
   - Stop timer.
   - Print: epoch index, predicted label, ground truth, correct/incorrect, latency.
6. **Print summary:** accuracy, mean/max latency.

---

## Step 4: Add Raw Data Loading Method to DataLoader

**File:** `src/data_loader.py`

Currently `get_epochs()` returns pre-extracted `mne.Epochs`. For streaming
playback we need access to the raw continuous data + events.

Add a method like:
```
get_raw_and_events(subject_id, task_type, paradigm)
  -> (mne.io.Raw, events_array, event_id_mapping)
```

This method should:
- Load and concatenate raw EDF files for the given runs.
- Rename annotations (same logic as existing).
- Apply bandpass filter.
- Select motor channels.
- Return the raw object, the events array, and the event_id dict.

---

## Step 5: Verify and Display Latency

In the streaming loop, wrap each prediction with timing:

```python
start = time.perf_counter()
prediction = pipeline.predict(epoch_data)
elapsed = time.perf_counter() - start
```

Print latency per epoch. Print summary stats (mean, max) at the end.
This proves the <2 second requirement to the evaluator.

---

## Summary of Files to Modify

| File | Change |
|---|---|
| `src/mybci.py` | Add `--model` and `--model-output` CLI options |
| `src/cli_commands/run.py` | Save model in train, rewrite predict as streaming playback |
| `src/data_loader.py` | Add `get_raw_and_events()` method |

No new files needed.
