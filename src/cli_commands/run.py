import click
from sklearn.model_selection import train_test_split
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
    verbose: bool = True,
) -> float:
    """
    Predict mode: train model and show per-epoch predictions.

    Displays prediction vs truth for each epoch in test set.

    Returns:
        Test accuracy as a float between 0 and 1.
    """
    # Load epochs
    epochs = data_loader.get_epochs(
        subject_id=subject_id,
        task_type=task_type,
        paradigm=task_paradigm,
    )

    # Create reverse mapping: event_id -> label name
    id_to_label = {v: k for k, v in epochs.event_id.items()}

    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # Build and train pipeline
    pipeline = construct_pipeline_from_config(config)
    if verbose:
        click.echo("Training model...")
    pipeline.fit(X_train, y_train)

    # Predict on each test epoch
    if verbose:
        click.echo()
    predictions = pipeline.predict(X_test)
    correct = 0

    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        is_correct = pred == truth
        if is_correct:
            correct += 1
        if verbose:
            pred_label = id_to_label.get(pred, str(pred))
            truth_label = id_to_label.get(truth, str(truth))
            click.echo(f"epoch {i:02d}: [{pred_label}] [{truth_label}] {is_correct}")

    # Calculate final accuracy
    accuracy = correct / len(y_test)
    if verbose:
        click.echo()
        click.echo(f"Accuracy: {accuracy:.4f}")

    return accuracy
