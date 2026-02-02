import click
import sys
from constants import TaskType, TaskParadigm
from data_loader import EEGDataLoader, EpochingConfig
from pipeline import BCIPipelineConfig
from cli_commands import run_train_mode, run_predict_mode, run_evaluate, run_visualize

DEFAULT_DATA_DIR = "../data/raw"


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """BCI Pipeline CLI for EEG Motor Imagery Classification."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('--subject', '-s', type=int, required=True,
              help='Subject number (1-109)')
@click.option('--task-type', '-t', 'task_type_str',
              type=click.Choice([t.value for t in TaskType]),
              required=True, help='Task type: motor_execution or motor_imagery')
@click.option('--task-paradigm', '-p', 'task_paradigm_str',
              type=click.Choice([p.value for p in TaskParadigm]),
              required=True, help='Task paradigm: left_right_hand or hands_feet')
@click.option('--mode', '-m', type=click.Choice(['train', 'predict']),
              default='train', help='Mode: train (CV + evaluation) or predict (per-epoch predictions)')
@click.option('--use-wavelets', '-w', is_flag=True,
              help='Use Haar wavelet transform (BONUS)')
@click.option('--data-dir', '-d', type=click.Path(exists=True),
              default=DEFAULT_DATA_DIR, help='Path to raw data directory')
@click.option('--cv-folds', type=int, default=5,
              help='Number of cross-validation folds')
@click.option('--test-size', type=float, default=0.2,
              help='Fraction of data for test set')
@click.option('--n-components', type=int, default=6,
              help='Number of CSP components')
@click.option('--t-min', type=float, default=0.5,
              help='Epoch start time relative to event (seconds)')
@click.option('--t-max', type=float, default=3.0,
              help='Epoch end time relative to event (seconds)')
@click.option('--l-freq', type=float, default=8.0,
              help='Low cutoff frequency for bandpass filter (Hz)')
@click.option('--h-freq', type=float, default=30.0,
              help='High cutoff frequency for bandpass filter (Hz)')
@click.option('--wavelet-level', type=int, default=8,
              help='Wavelet decomposition level (used with --use-wavelets)')
@click.option('--algorithm', '-a', type=click.Choice(['lda', 'logreg', 'svc']),
              default='lda', help='Classifier algorithm')
def run(subject, task_type_str, task_paradigm_str, mode, use_wavelets,
        data_dir, cv_folds, test_size, n_components,
        t_min, t_max, l_freq, h_freq, wavelet_level, algorithm):
    """
    Run training or prediction on a single subject.

    Examples:
        python mybci.py run -s 1 -t motor_imagery -p left_right_hand
        python mybci.py run -s 1 -t motor_imagery -p left_right_hand -m predict
    """
    # Convert string values to enums
    task_type = TaskType(task_type_str)
    task_paradigm = TaskParadigm(task_paradigm_str)
    subject_id = f"S{subject:03d}"

    click.echo(f"Running experiment for {subject_id}")
    click.echo(f"  Mode:       {mode}")
    click.echo(f"  Task Type:  {task_type.value}")
    click.echo(f"  Paradigm:   {task_paradigm.value}")
    click.echo(f"  Algorithm:  {algorithm}")
    click.echo(f"  Epoch:      [{t_min}, {t_max}] s")
    click.echo(f"  Filter:     [{l_freq}, {h_freq}] Hz")
    if use_wavelets:
        click.echo(f"  Wavelet:    level {wavelet_level}")
    click.echo()

    # Create epoching config
    epoching_config = EpochingConfig(
        t_min=t_min,
        t_max=t_max,
        l_freq=l_freq,
        h_freq=h_freq,
    )

    # Initialize data loader
    try:
        data_loader = EEGDataLoader(raw_data_dir=data_dir, epoching_config=epoching_config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Validate subject exists
    if subject_id not in data_loader.subject_ids:
        click.echo(f"Error: Subject {subject_id} not found in dataset", err=True)
        click.echo(f"Available subjects: {data_loader.subject_ids[0]} - {data_loader.subject_ids[-1]}", err=True)
        sys.exit(1)

    # Create pipeline config
    config = BCIPipelineConfig(
        cv_folds=cv_folds,
        test_size=test_size,
        n_csp_components=n_components,
        use_wavelet=use_wavelets,
        wavelet_level=wavelet_level,
        classifier_algorithm=algorithm,
    )

    # Dispatch to appropriate mode
    try:
        if mode == 'train':
            run_train_mode(data_loader, config, subject_id, task_type, task_paradigm)
        else:
            run_predict_mode(data_loader, config, subject_id, task_type, task_paradigm)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def evaluate():
    """Evaluate pipeline across all subjects and experiments."""
    run_evaluate()


@cli.command()
def visualize():
    """Visualize raw and filtered EEG data."""
    run_visualize()


if __name__ == "__main__":
    cli()
