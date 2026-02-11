import click
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from data_loader import EEGDataLoader
from pipeline import BCIPipelineConfig, construct_pipeline_from_config, train_and_evaluate_on_subject
from constants import TaskType, TaskParadigm


# Define the 4 experiments as (task_type, paradigm) pairs with experiment index
EXPERIMENTS: List[Tuple[int, TaskType, TaskParadigm, str]] = [
    (0, TaskType.MOTOR_EXECUTION, TaskParadigm.LEFT_RIGHT_HAND, "Motor Execution - Left/Right Hand"),
    (1, TaskType.MOTOR_IMAGERY, TaskParadigm.LEFT_RIGHT_HAND, "Motor Imagery - Left/Right Hand"),
    (2, TaskType.MOTOR_EXECUTION, TaskParadigm.HANDS_FEET, "Motor Execution - Hands/Feet"),
    (3, TaskType.MOTOR_IMAGERY, TaskParadigm.HANDS_FEET, "Motor Imagery - Hands/Feet"),
]


@dataclass
class ExperimentResults:
    """Results for a single experiment across all subjects."""
    experiment_id: int
    name: str
    task_type: TaskType
    paradigm: TaskParadigm
    subject_accuracies: Dict[str, float]  # subject_id -> accuracy

    @property
    def mean_accuracy(self) -> float:
        if not self.subject_accuracies:
            return 0.0
        return sum(self.subject_accuracies.values()) / len(self.subject_accuracies)

    @property
    def n_subjects(self) -> int:
        return len(self.subject_accuracies)


def run_evaluate(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    subject_ids: List[str] = None,
    verbose: bool = False,
) -> Tuple[List[ExperimentResults], float]:
    """
    Evaluate pipeline across all subjects and experiments.

    Runs the full evaluation matrix:
    - All subjects (S001-S109 by default)
    - All task types (motor_execution, motor_imagery)
    - All paradigms (left_right_hand, hands_feet)

    Args:
        data_loader: EEGDataLoader instance
        config: Pipeline configuration
        subject_ids: List of subject IDs to evaluate. If None, uses all subjects.
        verbose: If True, prints per-subject accuracy. Otherwise shows progress bar.

    Returns:
        Tuple of (list of ExperimentResults, overall mean accuracy)
    """
    if subject_ids is None:
        subject_ids = data_loader.subject_ids

    all_experiment_results: List[ExperimentResults] = []
    all_accuracies: List[float] = []

    for exp_id, task_type, paradigm, exp_name in EXPERIMENTS:
        if verbose:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Experiment {exp_id}: {exp_name}")
            click.echo(f"{'=' * 60}")

        subject_accuracies: Dict[str, float] = {}

        # Use progress bar when not verbose
        subject_iter = subject_ids
        if not verbose:
            subject_iter = tqdm(
                subject_ids,
                desc=f"Exp {exp_id}: {exp_name}",
                unit="subj",
            )

        for subject_id in subject_iter:
            try:
                pipeline = construct_pipeline_from_config(config)
                result = train_and_evaluate_on_subject(
                    data_loader=data_loader,
                    pipeline=pipeline,
                    subject_id=subject_id,
                    task_paradigm=paradigm,
                    task_type=task_type,
                    test_size=config.test_size,
                    cv_folds=config.cv_folds,
                    random_state=config.random_state,
                )
                accuracy = result.test_accuracy
                subject_accuracies[subject_id] = accuracy
                all_accuracies.append(accuracy)

                if verbose:
                    click.echo(f"experiment {exp_id}: subject {subject_id[1:]}: accuracy = {accuracy:.4f}")

            except Exception as e:
                if verbose:
                    click.echo(f"experiment {exp_id}: subject {subject_id[1:]}: FAILED - {e}", err=True)

        exp_results = ExperimentResults(
            experiment_id=exp_id,
            name=exp_name,
            task_type=task_type,
            paradigm=paradigm,
            subject_accuracies=subject_accuracies,
        )
        all_experiment_results.append(exp_results)

    # Calculate overall mean
    overall_mean = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0

    # Print summary (always shown)
    click.echo()
    click.echo("=" * 60)
    click.echo("Mean accuracy of the four different experiments for all subjects:")
    click.echo("=" * 60)

    for exp_results in all_experiment_results:
        click.echo(
            f"experiment {exp_results.experiment_id}: "
            f"accuracy = {exp_results.mean_accuracy:.4f} "
            f"({exp_results.n_subjects} subjects)"
        )

    click.echo()
    click.echo(f"Mean accuracy of 4 experiments: {overall_mean:.4f}")

    return all_experiment_results, overall_mean
