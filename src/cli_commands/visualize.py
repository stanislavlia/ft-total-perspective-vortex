import click
import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from data_loader import EEGDataLoader
from constants import TaskType, TaskParadigm, MOTOR_CHANNELS


def _load_raw_for_subject(
    data_loader: EEGDataLoader,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
) -> mne.io.Raw:
    """Load and concatenate raw EDF files for a subject, pick motor channels."""
    files = data_loader._get_edf_files_for_subject(subject_id, task_paradigm, task_type)
    if not files:
        raise ValueError(
            f"No files found for {subject_id} with "
            f"task_type={task_type.value}, paradigm={task_paradigm.value}"
        )

    raw_list = []
    for filepath in files:
        run_id = data_loader._extract_run_id(filepath)
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data_loader._rename_annotations(raw, run_id)
        raw_list.append(raw)

    raw_concat = mne.concatenate_raws(raw_list)
    raw_concat.pick_channels(MOTOR_CHANNELS, ordered=True)
    return raw_concat


def _plot_signal(ax, raw: mne.io.Raw, title: str, duration: float = 10.0, n_channels: int = 5):
    """Plot EEG time series on a given axes."""
    sfreq = raw.info['sfreq']
    n_samples = int(duration * sfreq)
    data = raw.get_data(start=0, stop=n_samples)
    times = np.arange(n_samples) / sfreq
    ch_names = raw.ch_names[:n_channels]

    for i, ch in enumerate(ch_names):
        offset = i * 0.0001  # vertical offset between channels
        ax.plot(times, data[i] + offset, linewidth=0.5, label=ch)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V) + offset")
    ax.legend(loc="upper right", fontsize=7)


def _plot_psd(ax, raw: mne.io.Raw, title: str, fmax: float = 60.0):
    """Plot power spectral density on a given axes."""
    spectrum = raw.compute_psd(method="welch", fmax=fmax, verbose=False)
    freqs = spectrum.freqs
    psds = spectrum.get_data()

    # Convert to dB
    psds_db = 10 * np.log10(psds)
    mean_psd = psds_db.mean(axis=0)

    ax.plot(freqs, mean_psd, linewidth=1.0, color="steelblue")
    ax.fill_between(
        freqs,
        psds_db.min(axis=0),
        psds_db.max(axis=0),
        alpha=0.2,
        color="steelblue",
        label="Channel range",
    )
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(loc="upper right", fontsize=7)


def run_visualize(
    data_loader: EEGDataLoader,
    subject_id: str,
    task_type: TaskType,
    task_paradigm: TaskParadigm,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    show_psd: bool = False,
    save_path: Optional[str] = None,
):
    """
    Visualize raw vs filtered EEG data.

    Plots before/after bandpass filtering. Optionally includes PSD plots.
    """
    click.echo(f"Loading raw data for {subject_id}...")
    raw = _load_raw_for_subject(data_loader, subject_id, task_type, task_paradigm)

    # Create filtered copy
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    click.echo(f"Plotting (filter: [{l_freq}, {h_freq}] Hz)...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    if show_psd:
        _plot_psd(axes[0], raw, title=f"{subject_id} — PSD (Raw)")
        _plot_psd(axes[1], raw_filtered, title=f"{subject_id} — PSD (Filtered [{l_freq}-{h_freq} Hz])")
    else:
        _plot_signal(axes[0], raw, title=f"{subject_id} — Raw EEG Signal")
        _plot_signal(axes[1], raw_filtered, title=f"{subject_id} — Filtered EEG Signal [{l_freq}-{h_freq} Hz]")

    fig.suptitle(
        f"EEG Visualization: {subject_id} — {task_type.value}, {task_paradigm.value}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        click.echo(f"Plot saved to {save_path}")
    else:
        plt.show()
