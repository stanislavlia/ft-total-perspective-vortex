# Updated README with Clarifications

# Total Perspective Vortex - Key Requirements

## Project Goal
Build a brain-computer interface that classifies EEG motor imagery data (imagining hand vs feet movements) achieving **minimum 60% accuracy** across all test subjects.

## Core Technical Requirements

### 1. **EEG Data Processing**
**Preprocessing Pipeline:**
- Parse EEG data from PhysioNet using MNE library (source: https://physionet.org/content/eegmmidb/1.0.0/)
- **Visualize raw data before and after filtering** (mandatory - checked in evaluation)
- Apply frequency band filtering (8-30 Hz recommended for motor imagery)
- Extract features or use raw filtered signals for CSP
- Use Fourier or wavelet transforms for spectral analysis (optional bonus)

**Data Structure:**
- 109 subjects, each with 14 runs (R01-R14)
- R01-R02: Baseline (not used for classification)
- R03-R14: 4 task types × 3 repetitions each
- **Combine 3 repetitions per task** for sufficient training data (~45 trials per experiment)

**Task Types:**
R03, R07, R11: Motor Execution - Left/Right Hand
R04, R08, R12: Motor Imagery - Left/Right Hand
R05, R09, R13: Motor Execution - Hands/Feet
R06, R10, R14: Motor Imagery - Hands/Feet

### 2. **Implement Dimensionality Reduction**

**IMPORTANT CLARIFICATION:**
- **Implementation requirement**: Understand and implement CSP algorithm structure
- **Allowed to use**: NumPy/SciPy functions for mathematical operations:
  - `np.linalg.eig()` for eigenvalue decomposition
  - `np.linalg.svd()` for singular value decomposition
  - `np.cov()` for covariance matrix estimation
- **Can use MNE's CSP**: `from mne.decoding import CSP` is acceptable
- **Focus**: Building complete BCI system, not reimplementing basic linear algebra

**Algorithm (CSP recommended):**
- Find transformation matrix W that maximizes class discrimination
- Projects signals to spatial patterns with maximum variance between classes
- Must integrate with sklearn using BaseEstimator and TransformerMixin classes

**Mathematical Goal:**
- Transform data: W^T × X = X_CSP
- Input: (n_trials, n_channels, n_times)
- Output: (n_trials, n_components) where n_components typically = 4

### 3. **Build Complete sklearn Pipeline**
**Pipeline Components:**
1. CSP for spatial filtering (4 components recommended)
2. LDA or other sklearn classifier
3. Optional: StandardScaler for normalization

**Critical Requirements:**
- Use sklearn's Pipeline object
- Apply cross-validation to **entire pipeline** (not just classifier)
- Predictions must complete within **2 seconds** of data receipt (for demo mode)
- Do NOT use mne-realtime library

**Example:**
```python
pipeline = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True)),
    ('lda', LinearDiscriminantAnalysis())
])
```

### 4. **Number of Experiments: 4 (NOT 6)**

**CLARIFICATION (Based on Checklist & Real Implementations):**
- **Checklist explicitly states**: "The mean of the resulting **four means** (corresponding to the **four types of experiment runs**)"
- **PDF mentions "6 experiments"**: This appears to be an **error or outdated version**
- **All successful implementations use 4 experiments**

**The 4 Experiments:**
```python
EXPERIMENTS = {
    0: {
        'name': 'Motor Execution - Left/Right Hand',
        'runs': ['R03', 'R07', 'R11'],
        'paradigm': 'left_right_hand',
        'type': 'motor_execution'
    },
    1: {
        'name': 'Motor Imagery - Left/Right Hand',
        'runs': ['R04', 'R08', 'R12'],
        'paradigm': 'left_right_hand',
        'type': 'motor_imagery'
    },
    2: {
        'name': 'Motor Execution - Hands/Feet',
        'runs': ['R05', 'R09', 'R13'],
        'paradigm': 'hands_feet',
        'type': 'motor_execution'
    },
    3: {
        'name': 'Motor Imagery - Hands/Feet',
        'runs': ['R06', 'R10', 'R14'],
        'paradigm': 'hands_feet',
        'type': 'motor_imagery'
    }
}
```

**Why 4 Makes Sense:**
- ✓ Matches the 4 task types in PhysioNet dataset
- ✓ Confirmed by evaluation checklist
- ✓ Used in all successful GitHub implementations
- ✓ Each experiment combines 3 repetitions for adequate sample size

### 5. **Command-Line Interface**

**NOTE:** The CLI format shown in PDF is **suggestive, not mandatory**. You can design your own interface.

**Example Interface (PDF Style):**
```bash
# Train on specific subject/task
python mybci.py 4 14 train
# Arguments: subject_num=4 (S004), task_num=14 (R14)
# Maps to task type: Motor Imagery Hands/Feet
# Actually uses: R06 + R10 + R14 (all 3 repetitions)

# Predict on specific subject/task
python mybci.py 4 14 predict
# Shows epoch-by-epoch predictions

# Full evaluation (no arguments)
python mybci.py
# Evaluates all 109 subjects × 4 experiments
```

**Alternative Interface (Used by Real Repos):**
```bash
python train.py --subject 1 --experiment imagery_left_right
python evaluate.py --all-subjects
```

**Task Number Mapping:**
When user provides task number (e.g., 14), map it to the **task type** and load **all 3 repetitions**:
- Task 14 (R14) → Motor Imagery Hands/Feet → Load R06, R10, R14
- Task 4 (R04) → Motor Imagery Left/Right → Load R04, R08, R12

### 6. **Training & Evaluation Protocol**

**Within-Subject Evaluation (Standard BCI Practice):**
- Each subject gets their own model
- NO cross-subject training/testing
- Split each subject's data independently (80% train / 20% test)

**For Each Subject:**
```python
# 1. Load subject's data for one experiment (e.g., R04+R08+R12)
epochs = load_and_concatenate_to_epochs(files)  # ~45 trials

# 2. Split within this subject
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Train: ~36 trials, Test: ~9 trials

# 3. Cross-validation on training set
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# 4. Train on full training set
pipeline.fit(X_train, y_train)

# 5. Evaluate on test set (never-learned data from SAME subject)
test_accuracy = pipeline.score(X_test, y_test)
```

**"Never-Learned Data" Means:**
- Test trials not in training set
- From the **same subject** (not different subjects)
- Completely held out during training

**Why Within-Subject:**
- EEG signals highly subject-specific (skull thickness, anatomy, placement)
- Cross-subject accuracy typically 15-20% lower
- Standard practice in BCI research and competitions

**Validation:**
- Apply `cross_val_score` on **entire pipeline** during training
- Use for hyperparameter tuning and overfitting detection
- Final evaluation on separate test set

**Performance Targets:**
- **≥60% mean accuracy** on test data (PDF requirement)
- **≥75% mean accuracy** (Checklist requirement - more strict)
- Test across **all 109 subjects**
- Evaluate on **4 experiments**
- Report individual and aggregate accuracies

### 7. **Output Format**

**Training Output (Per Subject/Task):**
```
Training S004 on hands_feet - motor_imagery
Using runs: ['R06', 'R10', 'R14']
Total epochs: 45
[0.6666 0.4444 0.4444 0.4444 0.4444 0.6666 0.8888 0.1111 0.7777 0.4444]
cross_val_score: 0.5333
Test accuracy: 0.6666
```

**Prediction Output (Per Subject/Task):**
```
epoch nb: [prediction] [truth] equal?
epoch 00: [2] [1] False
epoch 01: [1] [1] True
epoch 02: [2] [1] False
...
Accuracy: 0.6666
```

**Full Evaluation Output:**
```
experiment 0: subject 001: accuracy = 0.6
experiment 0: subject 002: accuracy = 0.8
...

Mean accuracy of the four different experiments for all 109 subjects:
experiment 0: accuracy = 0.6450
experiment 1: accuracy = 0.5890
experiment 2: accuracy = 0.7120
experiment 3: accuracy = 0.6210

Mean accuracy of 4 experiments: 0.6418
```

### 8. **Visualization Requirements (Mandatory)**

**From Subject PDF:**
> "You will have to write a script to visualize raw data then filter it to keep only useful frequency bands, and visualize again after this preprocessing."

**From Checklist:**
> "Check if the data were parsed then visualized with a script, showing raw and filtered data. The plots should look like what is shown in the video, the filtered signal being 'cleaner'."

**Required Visualizations:**
1. **Raw EEG signal** (before filtering) - shows noise, drift, artifacts
2. **Filtered EEG signal** (after 8-30 Hz bandpass) - cleaner, focused on motor imagery
3. **Comparison plot** - side-by-side or overlay showing improvement

**Implementation:**
```bash
python visualize.py 1 4  # Visualize S001, R04
```

**What to Show:**
- Time-domain plots (signal amplitude over time)
- Frequency-domain plots (Power Spectral Density) - optional
- Clear difference showing filtering effectiveness

## Technology Stack

**Required Libraries:**
- **MNE:** EEG data parsing and preprocessing
- **scikit-learn:** Pipeline, cross-validation, classification
- **NumPy/SciPy:** Mathematical operations
- **matplotlib:** Visualizations
- **Python 3.x:** Programming language

## Submission Requirements

**Repository Contents:**
- Python scripts for training, prediction, evaluation, and visualization
- **Do NOT include dataset** in repository (too large)
- Clear README with usage instructions
- Requirements.txt or environment.yml

**Required Scripts:**
1. Main program with 3 modes (train/predict/evaluate)
2. Visualization script for preprocessing demonstration
3. Helper modules for data loading and preprocessing

## Key Success Criteria

✓ Visualization of raw and filtered data  
✓ CSP dimensionality reduction properly integrated  
✓ Proper sklearn Pipeline with cross-validation  
✓ Within-subject evaluation across all 109 subjects  
✓ **4 experiments** (not 6) covering all task types  
✓ 60%+ accuracy on held-out test data  
✓ Combining 3 repetitions per task for adequate data  
✓ Clear command-line interface and output formatting  

## Common Misconceptions Clarified

**❌ WRONG:** Must implement CSP completely from scratch  
**✅ CORRECT:** Can use MNE's CSP, focus is on building complete BCI system

**❌ WRONG:** 6 experiments (PDF mentions this)  
**✅ CORRECT:** 4 experiments (confirmed by checklist and all real implementations)

**❌ WRONG:** Train on subjects 1-100, test on 101-109 (cross-subject)  
**✅ CORRECT:** Train and test on same subject with 80/20 split (within-subject)

**❌ WRONG:** Use only 1 run per task (e.g., just R14)  
**✅ CORRECT:** Combine all 3 repetitions per task (e.g., R06+R10+R14)

**❌ WRONG:** Evaluation on training data  
**✅ CORRECT:** Evaluation on held-out test set never seen during training

**❌ WRONG:** Cross-validation replaces test set  
**✅ CORRECT:** CV for hyperparameter tuning, separate test set for final evaluation

## Optional Bonuses

- Implement wavelet transforms for preprocessing
- Test multiple dimensionality reduction algorithms (PCA, ICA)
- Implement custom classifier
- Deep learning approaches (CNN, RNN)
- Test on additional EEG datasets
- Implement eigenvalue/SVD functions from scratch (hard)
- Advanced feature extraction methods

## Evaluation Criteria (From Checklist)

**Preprocessing (Mandatory):**
- ✓ Data parsed and loaded correctly
- ✓ Visualizations showing raw and filtered data
- ✓ Filtered signal visibly cleaner

**Implementation (Mandatory):**
- ✓ Dimensionality reduction algorithm integrated
- ✓ sklearn Pipeline with BaseEstimator/TransformerMixin
- ✓ Cross-validation on entire pipeline

**Performance (Mandatory):**
- ✓ Mean accuracy ≥ 60% (PDF) or ≥ 75% (Checklist)
- ✓ Evaluated on all subjects
- ✓ Results for 4 experiments reported

**Bonus Points:**
- Advanced preprocessing (wavelets)
- Custom classifier implementation
- Work on additional datasets
- Hyperparameter tuning

---

## References

- PhysioNet EEG Dataset: https://physionet.org/content/eegmmidb/1.0.0/
- MNE Documentation: https://mne.tools/
- CSP Algorithm: Ramoser et al. (1998), "Optimal spatial filtering of single trial EEG"
- Successful implementations: 
  - https://github.com/owalid/total-perspective-vortex
  - https://github.com/SpenderJ/Total-perspective-vortex
