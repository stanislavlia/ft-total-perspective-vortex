import time
import click
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import EEGDataLoader
from pipeline import (
    BCIPipelineConfig,
    construct_pipeline_from_config,
    train_and_evaluate_on_subject,
    load_pipeline,
    save_pipeline_to_file
)
from constants import TaskType, TaskParadigm

def run_train_mode(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
    save_model: bool,
    model_path: str
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

    if save_model:
        save_pipeline_to_file(
            pipeline=result.pipeline,
            epoching_config=data_loader.epoching_config,
            path=model_path,
        )


def run_predict_mode(
    data_loader: EEGDataLoader,
    config: BCIPipelineConfig,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
    model_path: str,
    wait: bool = False,
) -> float:
    """
    Predict mode: streaming playback using a pre-trained model.

    Loads a saved pipeline and walks through the raw EEG file
    chronologically, extracting and classifying epochs one by one
    as events are encountered — simulating real-time BCI inference.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    # Load pre-trained pipeline and its epoching config
    click.echo(f"Loading model from {model_path}")
    pipeline, saved_config = load_pipeline(model_path)

    # Override data loader's epoching config with the one used during training
    data_loader.epoching_config = saved_config
    click.echo(
        f"  Epoch: [{saved_config.t_min}, {saved_config.t_max}] s  "
        f"Filter: [{saved_config.l_freq}, {saved_config.h_freq}] Hz"
    )

    # Load raw continuous data + events
    raw, events, event_id = data_loader.get_raw_and_events(
        subject_id=subject_id,
        task_type=task_type,
        paradigm=task_paradigm,
    )
    sfreq = raw.info['sfreq']
    t_min = saved_config.t_min
    t_max = saved_config.t_max
    n_samples = int((t_max - t_min) * sfreq)

    # Reverse mapping: event code -> label name
    id_to_label = {v: k for k, v in event_id.items()}

    click.echo(
        f"Streaming {len(events)} epochs from {subject_id} "
        f"({task_type.value}, {task_paradigm.value})..."
    )
    click.echo()

    correct = 0
    latencies = []
    prev_event_time = None

    for i, event in enumerate(events):
        event_sample = event[0]
        truth = event[2]
        event_time = event_sample / sfreq

        # Simulate real-time pacing between events
        if wait and prev_event_time is not None:
            gap = event_time - prev_event_time
            if gap > 0:
                time.sleep(gap)
        prev_event_time = event_time

        # Extract epoch window from raw continuous data
        start_sample = int(event_sample + t_min * sfreq)
        end_sample = start_sample + n_samples
        epoch_data = raw.get_data(start=start_sample, stop=end_sample)
        epoch_data = epoch_data[np.newaxis, :, :]  # (1, n_channels, n_times)

        # Predict with timing
        t_start = time.perf_counter()
        prediction = pipeline.predict(epoch_data)[0]
        elapsed = time.perf_counter() - t_start

        latencies.append(elapsed)
        is_correct = prediction == truth
        if is_correct:
            correct += 1

        event_time = event_sample / sfreq
        pred_label = id_to_label.get(prediction, str(prediction))
        truth_label = id_to_label.get(truth, str(truth))
        click.echo(
            f"epoch {i:02d} [t={event_time:7.1f}s]: "
            f"predicted={pred_label:<12s} truth={truth_label:<12s} "
            f"correct={str(is_correct):<5s}  latency={elapsed:.4f}s"
        )

    accuracy = correct / len(events)
    click.echo()
    click.echo(f"Accuracy:     {accuracy:.4f}")
    click.echo(f"Mean latency: {np.mean(latencies):.4f}s")
    click.echo(f"Max latency:  {np.max(latencies):.4f}s")

    return accuracy
