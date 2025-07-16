"""Tests for the unified writer."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from viberl.utils.writer import UnifiedWriter


class TestUnifiedWriter:
    """Test cases for UnifiedWriter."""

    def test_initialization_tensorboard_only(self):
        """Test initialization with only TensorBoard enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            )

            assert writer.enable_tensorboard is True
            assert writer.enable_wandb is False
            assert writer.tb_writer is not None
            assert writer.wandb_run is None
            writer.close()

    def test_initialization_wandb_only(self):
        """Test initialization with only wandb enabled."""
        with tempfile.TemporaryDirectory() as tmpdir, patch('wandb.init') as mock_wandb_init:
            mock_wandb_init.return_value = None

            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=False,
                enable_wandb=True,
            )

            assert writer.enable_tensorboard is False
            assert writer.enable_wandb is True
            assert writer.tb_writer is None
            assert writer.wandb_run is None  # Mocked
            writer.close()

    def test_initialization_both_enabled(self):
        """Test initialization with both TensorBoard and wandb enabled."""
        with tempfile.TemporaryDirectory() as tmpdir, patch('wandb.init') as mock_wandb_init:
            mock_wandb_init.return_value = None

            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=True,
            )

            assert writer.enable_tensorboard is True
            assert writer.enable_wandb is True
            assert writer.tb_writer is not None
            assert writer.wandb_run is None  # Mocked
            writer.close()

    def test_log_scalar(self):
        """Test logging scalar values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            )

            # Should not raise any errors
            writer.log_scalar('test/scalar', 1.0, 0)
            writer.log_scalar('test/scalar', 2.0, 1)
            writer.close()

    def test_log_scalars(self):
        """Test logging multiple scalar values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            )

            scalars = {'loss': 0.5, 'accuracy': 0.95, 'reward': 10.0}
            writer.log_scalars(scalars, 0)
            writer.close()

    def test_log_histogram(self):
        """Test logging histogram data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            )

            data = np.random.randn(100)
            writer.log_histogram('test/histogram', data, 0)
            writer.close()

    def test_context_manager(self):
        """Test using writer as context manager."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            ) as writer,
        ):
            writer.log_scalar('test/context', 1.0, 0)
            assert writer.tb_writer is not None

    def test_close(self):
        """Test proper cleanup on close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=True,
                enable_wandb=False,
            )

            writer.close()
            # Should not raise any errors on close

    def test_log_dir_creation(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'new_dir' / 'logs'
            assert not log_path.exists()

            writer = UnifiedWriter(
                log_dir=str(log_path),
                enable_tensorboard=True,
                enable_wandb=False,
            )

            assert log_path.exists()
            writer.close()

    def test_wandb_config(self):
        """Test wandb configuration."""
        with tempfile.TemporaryDirectory() as tmpdir, patch('wandb.init') as mock_wandb_init:
            mock_wandb_init.return_value = None

            config = {'learning_rate': 0.001, 'batch_size': 32}
            writer = UnifiedWriter(
                log_dir=tmpdir,
                enable_tensorboard=False,
                enable_wandb=True,
                wandb_config=config,
                project_name='test_project',
                run_name='test_run',
            )

            mock_wandb_init.assert_called_once_with(
                project='test_project',
                name='test_run',
                config=config,
                dir=tmpdir,
            )
            writer.close()
