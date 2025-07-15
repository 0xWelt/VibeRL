"""Comprehensive tests for the Trainer class."""

import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from viberl.agents.reinforce import REINFORCEAgent
from viberl.trainer import Trainer
from viberl.utils.mock_env import MockEnv


class TestTrainerInitialization:
    """Test Trainer initialization and setup."""

    def test_trainer_initialization_basic(self):
        """Test basic Trainer initialization."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        trainer = Trainer(env=env, agent=agent)

        assert trainer.env == env
        assert trainer.agent == agent
        assert trainer.max_steps == 1000
        assert trainer.log_dir is None
        assert trainer.device.type in ['cpu', 'cuda']
        assert trainer.eval_env is None

    def test_trainer_initialization_with_eval_env(self):
        """Test Trainer initialization with eval_env parameter."""
        env = MockEnv(state_size=4)
        eval_env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        trainer = Trainer(env=env, agent=agent, eval_env=eval_env)

        assert trainer.eval_env == eval_env

    def test_trainer_initialization_with_log_dir(self):
        """Test Trainer initialization with log directory."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(env=env, agent=agent, log_dir=temp_dir)

            assert trainer.log_dir == temp_dir
            assert trainer.writer is not None

    def test_trainer_device_configuration(self):
        """Test device configuration in Trainer."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        # Test auto device
        trainer = Trainer(env=env, agent=agent, device='auto')
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert trainer.device.type == expected_device

        # Test specific device
        trainer = Trainer(env=env, agent=agent, device='cpu')
        assert trainer.device.type == 'cpu'


class TestTrainerTraining:
    """Test Trainer training functionality."""

    def test_train_basic(self):
        """Test basic training functionality."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        rewards = trainer.train(num_episodes=2, verbose=False)

        assert isinstance(rewards, list)
        assert len(rewards) == 2
        assert all(isinstance(r, int | float) for r in rewards)

    def test_train_with_evaluation(self):
        """Test training with evaluation."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        rewards = trainer.train(num_episodes=2, eval_interval=1, eval_episodes=1, verbose=False)

        assert len(rewards) == 2
        assert len(trainer.eval_rewards) > 0

    def test_train_with_tensorboard(self):
        """Test training with TensorBoard logging."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(env=env, agent=agent, log_dir=temp_dir, max_steps=10)
            rewards = trainer.train(num_episodes=2, verbose=False)

            assert len(rewards) == 2

    def test_train_with_render_interval(self):
        """Test training with rendering."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        with patch.object(env, 'render') as mock_render:
            trainer.train(num_episodes=2, render_interval=1, verbose=False)

            mock_render.assert_called()

    def test_train_interval_validation(self):
        """Test interval validation in training."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        # Test invalid save interval
        with pytest.raises(AssertionError):
            trainer.train(save_interval=3, eval_interval=2)

        # Test valid interval alignment
        rewards = trainer.train(num_episodes=2, eval_interval=1, save_interval=2, verbose=False)
        assert len(rewards) == 2


class TestTrainerEvaluation:
    """Test Trainer evaluation functionality."""

    def test_evaluate_basic(self):
        """Test basic evaluation functionality."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        scores, lengths = trainer.evaluate(num_episodes=2)

        assert isinstance(scores, list)
        assert isinstance(lengths, list)
        assert len(scores) == 2
        assert len(lengths) == 2

    def test_evaluate_with_rendering(self):
        """Test evaluation with rendering."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        with patch.object(env, 'render'):
            trainer.evaluate(num_episodes=1, render=True)

            # MockEnv might not have render, so this is expected behavior
            # Just ensure the function completes without error

    def test_evaluate_with_custom_env(self):
        """Test evaluation with custom evaluation environment."""
        train_env = MockEnv(state_size=4)
        eval_env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=train_env, agent=agent, eval_env=eval_env, max_steps=10)

        scores, lengths = trainer.evaluate(num_episodes=1)

        assert len(scores) == 1
        assert len(lengths) == 1


class TestTrainerEvalEnvFunctionality:
    """Test eval_env parameter functionality."""

    def test_eval_env_deepcopy_fallback(self):
        """Test deepcopy fallback for eval environment."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        # Should create a deepcopy for evaluation
        eval_env = trainer._create_eval_env()
        assert eval_env is not None
        eval_env.close()

    def test_eval_env_provided_directly(self):
        """Test when eval_env is provided directly."""
        env = MockEnv(state_size=4)
        eval_env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, eval_env=eval_env, max_steps=10)

        # Should use the provided eval_env
        created_eval_env = trainer._create_eval_env()
        assert created_eval_env == eval_env

    def test_eval_env_deepcopy_failure_handling(self):
        """Test handling when deepcopy fails."""

        class UncopyableEnv(MockEnv):
            def __deepcopy__(self, memo):  # noqa: ANN001
                raise RuntimeError('Cannot deepcopy')

        env = UncopyableEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        # Should fallback to using training env
        eval_env = trainer._create_eval_env()
        assert eval_env == env

    def test_eval_env_cleanup(self):
        """Test proper cleanup of evaluation environment."""
        env = MockEnv(state_size=4)
        eval_env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, eval_env=eval_env, max_steps=10)

        with patch.object(eval_env, 'close') as mock_close:
            trainer.evaluate(num_episodes=1)
            mock_close.assert_called_once()


class TestTrainerUtilities:
    """Test Trainer utility functions."""

    def test_preprocess_state(self):
        """Test state preprocessing."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent)

        # Test 1D state (no preprocessing needed)
        state_1d = np.array([1, 2, 3, 4])
        processed = trainer._preprocess_state(state_1d)
        assert np.array_equal(processed, state_1d)

        # Test 2D state (flattening needed)
        state_2d = np.array([[1, 2], [3, 4]])
        processed = trainer._preprocess_state(state_2d)
        expected = np.array([1, 2, 3, 4])
        assert np.array_equal(processed, expected)

    def test_save_checkpoint(self):
        """Test checkpoint saving functionality."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = temp_dir + '/test_models'

            # Mock agent save method
            with patch.object(agent, 'save') as mock_save:
                trainer._save_checkpoint(100, [10.0, 12.0], save_path)

                mock_save.assert_called_once()

    def test_log_progress(self):
        """Test progress logging."""
        env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)
        trainer = Trainer(env=env, agent=agent, max_steps=10)

        # Add some rewards for logging
        trainer.episode_rewards = [10.0, 12.0, 15.0]

        # Test logging with verbose=False
        trainer._log_progress(
            episode=100, num_episodes=1000, eval_rewards=[12.0, 15.0], eval_mean=13.5, verbose=False
        )

        # Should not raise any exceptions


class TestTrainerIntegration:
    """Integration tests for Trainer."""

    def test_full_training_cycle(self):
        """Test complete training cycle with all features."""
        env = MockEnv(state_size=4)
        eval_env = MockEnv(state_size=4)
        agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.001)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                env=env, agent=agent, eval_env=eval_env, log_dir=temp_dir, max_steps=10
            )

            rewards = trainer.train(
                num_episodes=3,
                eval_interval=1,
                eval_episodes=1,
                save_interval=2,
                save_path=temp_dir + '/models',
                verbose=False,
            )

            assert len(rewards) == 3
            assert len(trainer.eval_rewards) > 0
            assert trainer.writer is not None
