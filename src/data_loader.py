import mne
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from constants import MOTOR_CHANNELS, RUN_TYPE_TO_TASK, TaskParadigm, TaskType
from loguru import logger

mne.set_log_level('WARNING')


@dataclass
class EpochingConfig:
    t_min: float = 0.5
    t_max: float = 3.0
    select_channels: List[str] = field(default_factory=lambda: MOTOR_CHANNELS)
    apply_filter: bool = True
    l_freq: float = 8.0
    h_freq: float = 30.0
    baseline: Optional[tuple] = None


class EEGDataLoader:
    """
    Data loader for PhysioNet Motor Imagery EEG dataset.
    Returns MNE Epochs filtered by subject, task type, and paradigm.
    """

    def __init__(self, raw_data_dir: str, epoching_config: Optional[EpochingConfig] = None):
        self.raw_data_dir = raw_data_dir
        self.epoching_config = epoching_config or EpochingConfig()
        self._check_raw_data_dir()
        self._subject_ids = self._discover_subjects()

    def _check_raw_data_dir(self) -> None:
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        if not os.path.isdir(self.raw_data_dir):
            raise NotADirectoryError(f"Path is not a directory: {self.raw_data_dir}")
        if not os.listdir(self.raw_data_dir):
            raise ValueError(f"Raw data directory is empty: {self.raw_data_dir}")

    def _discover_subjects(self) -> List[str]:
        subjects = [
            item for item in sorted(os.listdir(self.raw_data_dir))
            if os.path.isdir(os.path.join(self.raw_data_dir, item)) and item.startswith('S')
        ]
        logger.info(f"Discovered {len(subjects)} subjects in {self.raw_data_dir}")
        return subjects

    @property
    def subject_ids(self) -> List[str]:
        return self._subject_ids

    @staticmethod
    def _extract_run_id(filepath: str) -> str:
        return 'R' + filepath.split('R')[-1].split('.')[0]

    def _get_runs_for_paradigm_and_task(
        self,
        paradigm: TaskParadigm,
        task_type: TaskType
    ) -> List[str]:
        matching_runs = [
            run_id for run_id, info in RUN_TYPE_TO_TASK.items()
            if info.get('paradigm') == paradigm.value
            and info.get('task_type') == task_type.value
        ]
        return sorted(matching_runs)

    def _get_edf_files_for_subject(
        self,
        subject_id: str,
        paradigm: TaskParadigm,
        task_type: TaskType
    ) -> List[str]:
        matching_runs = self._get_runs_for_paradigm_and_task(paradigm, task_type)
        files = [
            os.path.join(self.raw_data_dir, subject_id, f"{subject_id}{run_id}.edf")
            for run_id in matching_runs
        ]
        existing_files = [f for f in files if os.path.exists(f)]

        if len(existing_files) != len(files):
            missing = set(files) - set(existing_files)
            logger.warning(f"Missing files for {subject_id}: {missing}")

        return existing_files

    @staticmethod
    def _rename_annotations(raw: mne.io.Raw, run_id: str) -> Dict[str, int]:
        task_info = RUN_TYPE_TO_TASK[run_id]

        if task_info['labels'] is not None:
            annotation_mapping = {
                'T0': 'rest',
                'T1': task_info['labels']['T1'],
                'T2': task_info['labels']['T2']
            }
            raw.annotations.rename(annotation_mapping)

        events, event_dict = mne.events_from_annotations(raw)
        return event_dict

    def _load_single_file_to_epochs(self, filepath: str) -> mne.Epochs:
        run_id = self._extract_run_id(filepath)
        config = self.epoching_config

        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        event_dict = self._rename_annotations(raw, run_id)

        events, _ = mne.events_from_annotations(raw)
        event_id = {k: v for k, v in event_dict.items() if k != 'rest'}

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=config.t_min,
            tmax=config.t_max,
            baseline=config.baseline,
            picks=config.select_channels,
            preload=True,
            verbose=False
        )

        return epochs

    def get_epochs(
        self,
        subject_id: str,
        task_type: TaskType,
        paradigm: TaskParadigm
    ) -> mne.Epochs:
        """
        Load and return concatenated epochs for a subject.
        Applies bandpass filter if configured.
        """
        if subject_id not in self._subject_ids:
            raise ValueError(f"Unknown subject: {subject_id}")

        files = self._get_edf_files_for_subject(subject_id, paradigm, task_type)

        if not files:
            raise ValueError(
                f"No files found for {subject_id} with "
                f"task_type={task_type.value}, paradigm={paradigm.value}"
            )

        logger.debug(f"Loading {len(files)} files for {subject_id}")

        epochs_list = [self._load_single_file_to_epochs(f) for f in files]
        concatenated_epochs = mne.concatenate_epochs(epochs_list)

        if self.epoching_config.apply_filter:
            concatenated_epochs.filter(
                l_freq=self.epoching_config.l_freq,
                h_freq=self.epoching_config.h_freq,
                verbose=False
            )

        logger.info(
            f"Loaded {len(concatenated_epochs)} epochs for {subject_id} "
            f"({task_type.value}, {paradigm.value})"
        )

        return concatenated_epochs

    def get_epochs_for_subjects(
        self,
        subject_ids: List[str],
        task_type: TaskType,
        paradigm: TaskParadigm
    ) -> Dict[str, mne.Epochs]:
        """Load epochs for multiple subjects. Returns dict mapping subject_id to Epochs."""
        results = {}
        for subject_id in subject_ids:
            try:
                results[subject_id] = self.get_epochs(subject_id, task_type, paradigm)
            except Exception as e:
                logger.error(f"Failed to load epochs for {subject_id}: {e}")
        return results


if __name__ == "__main__":

    loader = EEGDataLoader(
        raw_data_dir="/home/stanislav/Desktop/ft-total-perspective-vortex/data/raw",
        epoching_config=EpochingConfig(
            t_min=0,
            t_max=2.5,
            apply_filter=True
        )
    )

    epochs = loader.get_epochs_for_subjects(subject_ids=["S001", "S009"], task_type=TaskType.MOTOR_IMAGERY, paradigm=TaskParadigm.LEFT_RIGHT_HAND)

    print(epochs)