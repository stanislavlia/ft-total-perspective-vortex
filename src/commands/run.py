import sys
import click
from data_loader import EEGDataLoader
from pipeline import (
    BCIPipelineConfig,
    construct_pipeline_from_config,
    train_and_evaluate_on_subject,
)
from constants import TaskType, TaskParadigm


def run_train_mode(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
):
    """
    Train mode: cross-validate and evaluate on held-out test set.

    Displays CV scores per fold and final test accuracy.
    """
    pipeline = construct_pipeline_from_config(config)

    click.echo("Training and evaluating...")
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

    # Display results
    click.echo()
    click.echo("=" * 50)
    click.echo(f"Results for {subject_id}")
    click.echo("=" * 50)
    click.echo(f"Train samples: {result.n_train_samples}")
    click.echo(f"Test samples:  {result.n_test_samples}")
    click.echo()
    click.echo("Cross-validation scores (per fold):")
    for i, score in enumerate(result.cv_scores):
        click.echo(f"  Fold {i + 1}: {score:.4f}")
    click.echo()
    click.echo(f"CV Mean:       {result.cv_mean:.4f} (+/- {result.cv_std:.4f})")
    click.echo(f"Test Accuracy: {result.test_accuracy:.4f}")


def run_predict_mode(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
):
    """
    Predict mode: train model and show per-epoch predictions.

    Displays prediction vs truth for each epoch in test set.
    """
    # TODO: Implement predict mode
    # 1. Load epochs
    # 2. Split into train/test
    # 3. Train pipeline on training set
    # 4. Predict on each test epoch
    # 5. Display: epoch XX: [pred] [truth] True/False
    # 6. Display final accuracy
    click.echo("Predict mode not yet implemented")
    sys.exit(1)
