"""
Experiment management utilities.

This module provides utilities for managing experiment directories,
tensorboard logging, and model checkpoints.
"""

from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """
    Manages experiment directories with automatic naming and organization.

    Creates experiment directories in the format:
    experiments/{experiment_name}_{timestamp}/
    ├── tb_logs/          # TensorBoard logs
    └── models/           # Saved model checkpoints
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = 'experiments',
        timestamp_format: str = '%Y%m%d_%H%M%S',
    ):
        """
        Initialize experiment manager.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
            timestamp_format: Format for timestamp in directory name
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.timestamp_format = timestamp_format

        # Create experiment directory name with timestamp
        timestamp = datetime.now().strftime(timestamp_format)
        self.experiment_dir = self.base_dir / f'{experiment_name}_{timestamp}'

        # Create subdirectories
        self.tb_logs_dir = self.experiment_dir / 'tb_logs'
        self.models_dir = self.experiment_dir / 'models'

        # Create directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create experiment directory structure."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.tb_logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_tb_logs_path(self) -> Path:
        """Get path to TensorBoard logs directory."""
        return self.tb_logs_dir

    def get_models_path(self) -> Path:
        """Get path to models directory."""
        return self.models_dir

    def get_experiment_path(self) -> Path:
        """Get path to experiment directory."""
        return self.experiment_dir

    def save_model(self, model_name: str) -> Path:
        """
        Get full path for saving a model.

        Args:
            model_name: Name of the model file (without extension)

        Returns:
            Full path for model file
        """
        return self.models_dir / f'{model_name}.pth'

    def list_experiments(self) -> list[Path]:
        """List all existing experiments."""
        if not self.base_dir.exists():
            return []

        experiments = [
            exp_dir
            for exp_dir in self.base_dir.iterdir()
            if exp_dir.is_dir() and self.experiment_name in exp_dir.name
        ]

        return sorted(experiments, reverse=True)

    def get_latest_experiment(self) -> Path | None:
        """Get the latest experiment directory."""
        experiments = self.list_experiments()
        return experiments[0] if experiments else None

    def print_experiment_info(self) -> None:
        """Print information about the current experiment."""
        print(f'Experiment: {self.experiment_name}')
        print(f'Directory: {self.experiment_dir}')
        print(f'TensorBoard logs: {self.tb_logs_dir}')
        print(f'Models: {self.models_dir}')

    @staticmethod
    def create_from_existing(
        experiment_path: str | Path,
    ) -> 'ExperimentManager':
        """
        Create ExperimentManager from existing experiment directory.

        Args:
            experiment_path: Path to existing experiment directory

        Returns:
            ExperimentManager instance
        """
        experiment_path = Path(experiment_path)
        experiment_name = experiment_path.name.rsplit('_', 1)[0]

        # Create manager but don't create directories since they exist
        manager = ExperimentManager(experiment_name, str(experiment_path.parent))
        return manager


def create_experiment(
    experiment_name: str,
    base_dir: str = 'experiments',
    print_info: bool = True,
) -> ExperimentManager:
    """
    Convenience function to create a new experiment.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments
        print_info: Whether to print experiment info

    Returns:
        ExperimentManager instance
    """
    manager = ExperimentManager(experiment_name, base_dir)

    if print_info:
        manager.print_experiment_info()

    return manager
