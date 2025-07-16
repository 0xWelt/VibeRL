"""
Experiment management utilities.

This module provides utilities for managing experiment directories,
tensorboard logging, and model checkpoints.
"""

from datetime import datetime
from pathlib import Path

from loguru import logger


class ExperimentManager:
    """
    Manages experiment directories with automatic naming and organization.

    Creates experiment directories in the format:
    experiments/{experiment_name}_{timestamp}/
    ├── tb_logs/          # TensorBoard logs
    ├── models/           # Saved model checkpoints
    └── training.log      # Training log file
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

        # Core experiment directory
        self.exp_dir = self.experiment_dir

        # Standardized subdirectory paths
        self.tb_logs_dir = self.exp_dir / 'tb_logs'
        self.models_dir = self.exp_dir / 'models'

        # Create directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create experiment directory structure."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.tb_logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_tb_logs_path(self) -> Path:
        """Get path to TensorBoard logs directory."""
        return self.tb_logs_dir

    def get_models_path(self) -> Path:
        """Get path to models directory."""
        return self.models_dir

    def get_training_log_path(self) -> Path:
        """Get path to training log file."""
        return self.experiment_dir / 'training.log'

    def configure_file_logging(self, log_level: str = 'INFO') -> None:
        """
        Configure loguru to log to training.log file.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        log_file = self.get_training_log_path()

        # Remove any existing file handlers to avoid duplicates
        logger.remove()

        # Add console handler
        logger.add(
            lambda msg: print(msg, end=''),
            level=log_level,
            format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
        )

        # Add file handler
        logger.add(
            str(log_file),
            level=log_level,
            format='{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}',
            rotation='10 MB',  # Rotate at 10MB
            retention='10 days',  # Keep logs for 10 days
        )

    def log_command_line_args(self, args: object) -> None:
        """
        Log command line arguments to training.log.

        Args:
            args: Parsed argparse arguments object
        """
        logger.info('=' * 80)
        logger.info('TRAINING SESSION STARTED')
        logger.info('=' * 80)

        # Log command line arguments
        logger.info('Command Line Arguments:')
        for arg_name, arg_value in vars(args).items():
            logger.info(f'  {arg_name}: {arg_value}')

        logger.info('=' * 80)
        logger.info(f'Experiment Directory: {self.experiment_dir}')
        logger.info('=' * 80)

    def get_experiment_path(self) -> Path:
        """Get path to experiment directory."""
        return self.exp_dir

    def get_exp_dir(self) -> Path:
        """Get path to experiment directory (alias for get_experiment_path)."""
        return self.exp_dir

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
        logger.info(f'Experiment: {self.experiment_name}')
        logger.info(f'Directory: {self.exp_dir}')
        logger.info(f'TensorBoard logs: {self.tb_logs_dir}')
        logger.info(f'Models: {self.models_dir}')
        logger.info(f'Training log: {self.get_training_log_path()}')

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
