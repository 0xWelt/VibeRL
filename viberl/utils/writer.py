"""Unified logging writer for TensorBoard and Weights & Biases."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from typing import Self


# Import types for TYPE_CHECKING only
if TYPE_CHECKING:
    import types


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from torch.utils.tensorboard import SummaryWriter


if TYPE_CHECKING:
    import types


class UnifiedWriter:
    """Unified writer supporting TensorBoard and Weights & Biases logging."""

    def __init__(
        self,
        log_dir: str,
        project_name: str = 'VibeRL',
        run_name: str | None = None,
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize unified writer.

        Args:
            log_dir: Directory for TensorBoard logs
            project_name: Name of the wandb project
            run_name: Name for wandb run
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_wandb: Whether to enable Weights & Biases logging
            wandb_config: Configuration dict for wandb
        """
        self.log_dir = log_dir
        self.enable_tensorboard = enable_tensorboard
        self.enable_wandb = enable_wandb

        # Initialize TensorBoard writer
        self.tb_writer: SummaryWriter | None = None
        if enable_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir)

        # Initialize Weights & Biases
        self.wandb_run: Any | None = None
        if enable_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError(
                    'wandb is not installed. Please install it with: pip install wandb'
                )
            if wandb_config is None:
                wandb_config = {}
            self.wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config=wandb_config,
                dir=log_dir,
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to both TensorBoard and wandb."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            wandb.log({tag: value}, step=step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log multiple scalar values at once."""
        if self.tb_writer is not None:
            for tag, value in scalars.items():
                self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            wandb.log(scalars, step=step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log histogram data."""
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
        if self.wandb_run is not None:
            wandb.log({f'{tag}_hist': wandb.Histogram(values)}, step=step)

    def close(self) -> None:
        """Close all writers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None and WANDB_AVAILABLE:
            wandb.finish()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
