# Total Perspective Vortex - Key Requirements

## Project Goal
Build a brain-computer interface that classifies EEG motor imagery data (imagining hand vs feet movements) achieving **minimum 60% accuracy** across all test subjects.

## Core Technical Requirements

### 1. **EEG Data Processing**
**Preprocessing Pipeline:**
- Parse EEG data from PhysioNet using MNE library (source: https://physionet.org/content/eegmmidb/1.0.0/)
- Visualize raw data before and after filtering
- Apply frequency band filtering to retain useful signals
- Extract features (e.g., signal power by frequency and channel)
- Use Fourier or wavelet transforms for spectral analysis

**Data Structure:**
- Input signals: R^(channels × time)
- Feature matrix: R^(d × N) where d = channels × time
- N = number of events across all classes

### 2. **Implement Custom Dimensionality Reduction**
**Algorithm (CSP recommended):**
- Code the algorithm **from scratch** (not using sklearn's version)
- Find transformation matrix W that projects data to most discriminative features
- CSP maximizes variance between different motor imagery classes
- Must integrate with sklearn using BaseEstimator and TransformerMixin classes

**Mathematical Goal:**
- Transform data: W^T × X = X_CSP
- Projects data onto axes expressing maximum class separation

**Allowed Tools:**
- NumPy/SciPy functions ONLY for:
  - Eigenvalue decomposition
  - Singular value decomposition  
  - Covariance matrix estimation

### 3. **Build Complete sklearn Pipeline**
**Pipeline Components:**
1. Your custom dimensionality reduction (CSP/PCA/ICA)
2. sklearn classifier of your choice
3. Data streaming simulation for "real-time" processing

**Critical Requirements:**
- Use sklearn's Pipeline object
- Predictions must complete within **2 seconds** of data receipt
- Do NOT use mne-realtime library
- Pipeline must handle streaming data chunks

### 4. **Create Command-Line Interface**

**Training Mode:**
```bash
python mybci.py 4 14 train
# Output: cross-validation scores and mean accuracy
```

**Prediction Mode:**
```bash
python mybci.py 4 14 predict
# Output: epoch-by-epoch predictions vs ground truth
# Shows: epoch number, prediction, truth, match status
# Final accuracy for this run
```

**Full Evaluation Mode:**
```bash
python mybci.py
# Output: accuracy for each subject in each experiment type
# Mean accuracy per experiment (6 total)
# Overall mean accuracy across all experiments
```

### 5. **Training & Evaluation Protocol**

**Data Splitting:**
- Training set: ~60-70% for model training
- Validation set: Used during cross-validation
- Test set: ~20-30% **never-before-seen data**
- Use different splits each time to avoid overfitting

**Validation:**
- Apply `cross_val_score` on the **entire pipeline** (not just classifier)
- Implement k-fold cross-validation for parameter tuning

**Performance Targets:**
- **≥60% mean accuracy** on test data
- Test across **all 109 subjects**
- Evaluate on **6 different experiment types**
- Report individual and aggregate accuracies

### 6. **Output Examples**

**Training Output:**
```
[0.6666 0.4444 0.4444 0.4444 0.4444 0.6666 0.8888 0.1111 0.7777 0.4444]
cross_val_score: 0.5333
```

**Prediction Output:**
```
epoch 00: [2] [1] False
epoch 01: [1] [1] True
...
Accuracy: 0.6666
```

**Full Evaluation Output:**
```
experiment 0: subject 001: accuracy = 0.6
experiment 0: subject 002: accuracy = 0.8
...
Mean accuracy of 6 experiments: 0.6261
```

## Technology Stack

**Required Libraries:**
- **MNE:** EEG data parsing and preprocessing
- **scikit-learn:** Pipeline, cross-validation, classification
- **NumPy/SciPy:** Mathematical operations (limited use)
- **Python:** Programming language

## Submission Requirements

**Repository Contents:**
- Python scripts only (training, prediction, evaluation)
- **Do NOT include dataset** in repository
- Submit via Git repository

**Required Scripts:**
1. Main program: `mybci.py` with 3 modes
2. Custom dimensionality reduction implementation
3. Pipeline configuration

## Key Success Criteria

✓ Custom dimensionality reduction algorithm implemented from scratch  
✓ Proper sklearn Pipeline integration  
✓ Real-time prediction capability (<2 seconds)  
✓ Cross-validation on complete pipeline  
✓ 60%+ accuracy on unseen test data  
✓ Works across multiple subjects and experiment types

## Optional Bonuses
- Implement wavelet transforms for better preprocessing
- Code custom classifier
- Implement eigenvalue/SVD functions from scratch
- Test on additional datasets

---
# Subject-Specific BCI Model Strategy

## Approach
We employ a **within-subject evaluation framework** with subject-specific models, which is the standard approach in motor imagery BCI research.

## Model Architecture
- **Pipeline**: CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis)
- **One model per (subject, experiment) pair**: 109 subjects × 6 experiments = 654 models

## Data Splitting
- **Independent splits per subject**: Each subject's data is split separately (typically 80% train / 20% test)
- **No cross-subject mixing**: Training and testing data always come from the same subject
- **Never-learned data**: Test set contains trials never seen during training

## Evaluation Process
1. For each subject (1-109):
   - For each experiment (0-5):
     - Load subject's data for this experiment
     - Create train/test split within this subject
     - Train CSP+LDA pipeline on training set
     - Evaluate on test set
     - Record accuracy

2. Final metric: Mean accuracy across all 654 evaluations
3. Success criterion: Mean accuracy ≥ 60%

## Rationale
- **High inter-subject variability**: EEG patterns differ dramatically between individuals due to anatomical differences, electrode placement, and neural signatures
- **Performance considerations**: Cross-subject models typically show 15-20% accuracy degradation compared to within-subject
- **Real-world applicability**: Practical BCI systems require user calibration (5-10 minutes) before use
- **Standard practice**: BCI competitions and benchmarks use within-subject evaluation