from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
import numpy as np
from csp import CommonSpatialPattern
from constants import Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_loader import EEGDataLoader
from constants import TaskParadigm, TaskType
from loguru import logger

@dataclass
class BCIPipelineConfig:
    cv_folds: int = 5
    test_size: float = 0.2
    n_csp_components: int = 6
    classifier_algorithm: str = "lda"  # other options: logreg, svc
    random_state: int = 42


@dataclass
class SubjectEvaluationResult:
    """Results from evaluating pipeline on a single subject."""
    subject_id: str
    cv_scores: List[float]  # Score for each fold
    cv_mean: float
    cv_std: float
    test_accuracy: float
    n_train_samples: int
    n_test_samples: int


def construct_pipeline_from_config(config: BCIPipelineConfig) -> Pipeline:

    #validate
    assert config.cv_folds >= 2
    assert config.n_csp_components > 0
    assert config.test_size < 1 and config.test_size > 0
    assert config.classifier_algorithm in [c.value for c in Classifiers]


    pipeline_components = []

    pipeline_components.append(
        ("csp", CommonSpatialPattern(n_components=config.n_csp_components))
    )

    #FOR NOW USE ONLY LDA
    pipeline_components.append(
        ("lda",  LinearDiscriminantAnalysis())
    )

    return Pipeline(steps=pipeline_components)


    
def train_and_evaluate_on_subject(
    data_loader: EEGDataLoader,
    pipeline: Pipeline,
    subject_id: str,
    task_paradigm: TaskParadigm = TaskParadigm.LEFT_RIGHT_HAND,
    task_type: TaskType = TaskType.MOTOR_IMAGERY,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> SubjectEvaluationResult:
    """
    Train and evaluate a pipeline on a single subject.

    Args:
        data_loader: EEGDataLoader instance for loading subject data
        pipeline: sklearn Pipeline to train and evaluate
        subject_id: Subject identifier (e.g., "S001")
        task_paradigm: LEFT_RIGHT_HAND or HANDS_FEET
        task_type: MOTOR_EXECUTION or MOTOR_IMAGERY
        test_size: Fraction of data to use for testing
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        SubjectEvaluationResult with CV scores per fold and test accuracy
    """
    # Load epochs for this subject
    epochs = data_loader.get_epochs(
        subject_id=subject_id,
        task_type=task_type,
        paradigm=task_paradigm
    )

    # Extract data and labels
    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]  # Labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.debug(
        f"{subject_id}: {len(X_train)} train samples, {len(X_test)} test samples"
    )

    # Cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds)

    # Train on full training set and evaluate on test set
    pipeline.fit(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)

    return SubjectEvaluationResult(
        subject_id=subject_id,
        cv_scores=cv_scores.tolist(),
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        test_accuracy=float(test_accuracy),
        n_train_samples=len(X_train),
        n_test_samples=len(X_test)
    )


def run_experiment(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    task_paradigm: TaskParadigm,
    task_type: TaskType,
    subject_ids: List[str] = None,
) -> List[SubjectEvaluationResult]:
    """
    Run a full experiment across multiple subjects.

    Args:
        data_loader: EEGDataLoader instance
        config: Pipeline configuration
        task_paradigm: LEFT_RIGHT_HAND or HANDS_FEET
        task_type: MOTOR_EXECUTION or MOTOR_IMAGERY
        subject_ids: List of subject IDs to evaluate. If None, uses all subjects.

    Returns:
        List of SubjectEvaluationResult for each subject
    """
    if subject_ids is None:
        subject_ids = data_loader.subject_ids

    results = []

    for subject_id in tqdm(subject_ids, desc=f"{task_paradigm.value}/{task_type.value}"):
        try:
            # Create fresh pipeline for each subject
            pipeline = construct_pipeline_from_config(config)

            result = train_and_evaluate_on_subject(
                data_loader=data_loader,
                pipeline=pipeline,
                subject_id=subject_id,
                task_paradigm=task_paradigm,
                task_type=task_type,
                test_size=config.test_size,
                cv_folds=config.cv_folds,
                random_state=config.random_state,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate {subject_id}: {e}")
            continue

    return results


def summarize_experiment_results(results: List[SubjectEvaluationResult]) -> dict:
    """
    Summarize results from an experiment.

    Returns dict with:
        - mean_cv_score: Mean of all subjects' CV means
        - mean_test_accuracy: Mean of all subjects' test accuracies
        - std_test_accuracy: Std of test accuracies across subjects
        - n_subjects: Number of successfully evaluated subjects
    """
    if not results:
        return {
            "mean_cv_score": 0.0,
            "mean_test_accuracy": 0.0,
            "std_test_accuracy": 0.0,
            "n_subjects": 0
        }

    cv_means = [r.cv_mean for r in results]
    test_accs = [r.test_accuracy for r in results]

    return {
        "mean_cv_score": float(np.mean(cv_means)),
        "mean_test_accuracy": float(np.mean(test_accs)),
        "std_test_accuracy": float(np.std(test_accs)),
        "n_subjects": len(results)
    }

