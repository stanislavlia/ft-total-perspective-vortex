# CLI Reference

All commands are run from the `src/` directory:

```bash
cd src
python mybci.py <command> [options]
```

---

## `run` — Train or Predict on a Single Subject

### Train Mode

Train a CSP+LDA pipeline on a single subject with cross-validation and held-out test evaluation.

```bash
python mybci.py run -s <subject> -t <task_type> -p <paradigm> -m train [options]
```

**Required options:**
| Option | Short | Description |
|---|---|---|
| `--subject` | `-s` | Subject number (1-109) |
| `--task-type` | `-t` | `motor_execution` or `motor_imagery` |
| `--task-paradigm` | `-p` | `left_right_hand` or `hands_feet` |

**Optional:**
| Option | Default | Description |
|---|---|---|
| `--mode` / `-m` | `train` | `train` or `predict` |
| `--data-dir` / `-d` | `../data/raw` | Path to raw EEG data |
| `--cv-folds` | `5` | Number of cross-validation folds |
| `--test-size` | `0.2` | Fraction of data for test set |
| `--n-components` | `6` | Number of CSP components (must be even) |
| `--t-min` | `0.5` | Epoch start time relative to event (seconds) |
| `--t-max` | `3.0` | Epoch end time relative to event (seconds) |
| `--l-freq` | `8.0` | Bandpass filter low cutoff (Hz) |
| `--h-freq` | `30.0` | Bandpass filter high cutoff (Hz) |
| `--algorithm` / `-a` | `lda` | Classifier: `lda`, `logreg`, or `svc` |
| `--save-model` | off | Save trained pipeline to disk |
| `--model-path` | — | Path for saved model (.joblib) |

**Examples:**

```bash
# Basic training with default parameters
python mybci.py run -s 1 -t motor_imagery -p left_right_hand

# Train with custom epoch window and save the model
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m train \
  --t-min 0.5 --t-max 2.5 --save-model --model-path models/S001_mi_lr.joblib

# Train motor execution, hands vs feet
python mybci.py run -s 42 -t motor_execution -p hands_feet

# Train with 4 CSP components and wider filter
python mybci.py run -s 10 -t motor_imagery -p left_right_hand \
  --n-components 4 --l-freq 4.0 --h-freq 40.0
```

### Predict Mode (Streaming Playback)

Load a pre-trained model and stream through the raw EEG file epoch-by-epoch, classifying each event as it is encountered.

```bash
python mybci.py run -s <subject> -t <task_type> -p <paradigm> -m predict \
  --model-path <path_to_model>
```

`--model-path` is **required** in predict mode. The epoching config (t_min, t_max, filter frequencies) is loaded from the saved model file — CLI defaults are ignored.

**Additional predict option:**
| Option | Description |
|---|---|
| `--wait` | Sleep real time between events to simulate live playback |

**Examples:**

```bash
# Train on subject 1, then predict on the same subject
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m train \
  --save-model --model-path models/S001_mi_lr.joblib
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m predict \
  --model-path models/S001_mi_lr.joblib

# Cross-subject: train on subject 52, predict on subject 1
python mybci.py run -s 52 -t motor_imagery -p left_right_hand -m train \
  --save-model --model-path models/S052_mi_lr.joblib
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m predict \
  --model-path models/S052_mi_lr.joblib

# Predict with real-time pacing (waits between events)
python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m predict \
  --model-path models/S001_mi_lr.joblib --wait
```

---

## `evaluate` — Full Evaluation Across Subjects and Experiments

Run the complete evaluation matrix: all subjects (or a subset) across 4 experiment types (motor_execution/imagery x left_right_hand/hands_feet).

```bash
python mybci.py evaluate [options]
```

**Options:**
| Option | Short | Default | Description |
|---|---|---|---|
| `--data-dir` | `-d` | `../data/raw` | Path to raw EEG data |
| `--subjects` | `-s` | all 109 | Subject selection (see below) |
| `--n-components` | | `6` | CSP components |
| `--test-size` | | `0.2` | Test set fraction |
| `--t-min` | | `0.5` | Epoch start (seconds) |
| `--t-max` | | `3.0` | Epoch end (seconds) |
| `--l-freq` | | `8.0` | Filter low cutoff (Hz) |
| `--h-freq` | | `30.0` | Filter high cutoff (Hz) |
| `--verbose` | `-v` | off | Print per-subject accuracy |

**Subject selection formats:**
- Comma-separated: `--subjects 1,2,3`
- Range: `--subjects 1-10`
- Mixed: `--subjects 1,3-5,10`

**Examples:**

```bash
# Evaluate all 109 subjects (takes a while)
python mybci.py evaluate

# Quick test on a few subjects
python mybci.py evaluate --subjects 1,2,3

# Evaluate subjects 1-10 with verbose output
python mybci.py evaluate --subjects 1-10 --verbose

# Custom parameters
python mybci.py evaluate --subjects 1-20 --n-components 4 --test-size 0.3
```

---

## `visualize` — Plot Raw vs Filtered EEG Data

Show before/after comparison of EEG signal with bandpass filtering. Use `--psd` to show power spectral density instead.

```bash
python mybci.py visualize -s <subject> -t <task_type> -p <paradigm> [options]
```

**Required options:**
| Option | Short | Description |
|---|---|---|
| `--subject` | `-s` | Subject number (1-109) |
| `--task-type` | `-t` | `motor_execution` or `motor_imagery` |
| `--task-paradigm` | `-p` | `left_right_hand` or `hands_feet` |

**Optional:**
| Option | Default | Description |
|---|---|---|
| `--data-dir` / `-d` | `../data/raw` | Path to raw EEG data |
| `--l-freq` | `8.0` | Filter low cutoff (Hz) |
| `--h-freq` | `30.0` | Filter high cutoff (Hz) |
| `--psd` | off | Show PSD plots instead of signal plots |
| `--save` | — | Save plot to file instead of displaying |

**Examples:**

```bash
# Show raw vs filtered EEG signal (opens matplotlib window)
python mybci.py visualize -s 1 -t motor_imagery -p left_right_hand

# Show PSD before/after filtering
python mybci.py visualize -s 1 -t motor_imagery -p left_right_hand --psd

# Custom filter range
python mybci.py visualize -s 1 -t motor_imagery -p left_right_hand --l-freq 4.0 --h-freq 40.0

# Save signal plot to file
python mybci.py visualize -s 1 -t motor_imagery -p left_right_hand --save plots/S001_signal.png

# Save PSD plot to file
python mybci.py visualize -s 1 -t motor_imagery -p left_right_hand --psd --save plots/S001_psd.png
```
