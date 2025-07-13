"""
Tests for experiment manager utility.

Tests focus on experiment tracking and management functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest

from viberl.utils.experiment_manager import ExperimentManager, create_experiment


class TestExperimentManager:
    """Test ExperimentManager functionality."""

    def test_experiment_manager_creation(self):
        """Test ExperimentManager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager('test', base_dir=temp_dir)
            assert manager.experiment_name == 'test'
            assert temp_dir in str(manager.base_dir)

    def test_experiment_manager_directory_creation(self):
        """Test ExperimentManager creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager('test', base_dir=temp_dir)

            assert os.path.exists(manager.get_experiment_path())
            assert os.path.exists(manager.get_tb_logs_path())
            assert os.path.exists(manager.get_models_path())

    def test_experiment_manager_paths(self):
        """Test ExperimentManager provides correct paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager('test', base_dir=temp_dir)

            experiment_path = manager.get_experiment_path()
            tb_path = manager.get_tb_logs_path()
            models_path = manager.get_models_path()

            assert isinstance(experiment_path, Path)
            assert isinstance(tb_path, Path)
            assert isinstance(models_path, Path)
            assert 'tb_logs' in str(tb_path)
            assert 'models' in str(models_path)

    def test_experiment_manager_save_model(self):
        """Test ExperimentManager model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager('test', base_dir=temp_dir)

            model_path = manager.save_model('test_model')
            assert model_path.suffix == '.pth'
            assert 'test_model' in str(model_path)

    def test_experiment_manager_list_experiments(self):
        """Test ExperimentManager can list experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create experiment
            manager1 = ExperimentManager('exp1', base_dir=temp_dir)
            _ = ExperimentManager('exp2', base_dir=temp_dir)  # Create for directory

            experiments = manager1.list_experiments()

            # Should have at least 1 experiment (manager1)
            assert len(experiments) >= 1
            assert any('exp1' in str(exp) for exp in experiments)

    def test_experiment_manager_latest_experiment(self):
        """Test ExperimentManager can get latest experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create experiments with some time separation
            manager1 = ExperimentManager('test1', base_dir=temp_dir)
            _ = ExperimentManager('test2', base_dir=temp_dir)  # Create for directory

            latest = manager1.get_latest_experiment()

            assert latest is not None
            assert isinstance(latest, Path)

    def test_create_experiment_convenience(self):
        """Test convenience function for creating experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_experiment('convenience_test', base_dir=temp_dir, print_info=False)

            assert isinstance(manager, ExperimentManager)
            assert manager.experiment_name == 'convenience_test'

    def test_create_from_existing(self):
        """Test creating ExperimentManager from existing experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create original experiment
            original = ExperimentManager('existing', base_dir=temp_dir)

            # Create from existing
            from_existing = ExperimentManager.create_from_existing(original.get_experiment_path())

            assert isinstance(from_existing, ExperimentManager)
            assert 'existing' in str(from_existing.experiment_name)

    @pytest.mark.parametrize('experiment_name', ['test1', 'experiment_2', 'my_experiment'])
    def test_experiment_manager_various_names(self, experiment_name: str) -> None:
        """Test ExperimentManager with various experiment names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(experiment_name, base_dir=temp_dir)

            assert manager.experiment_name == experiment_name
            assert os.path.exists(manager.get_experiment_path())
