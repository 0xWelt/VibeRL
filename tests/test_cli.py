"""Tests for VibeRL CLI functions."""

import os
import sys
from unittest.mock import Mock, patch

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viberl.cli import demo_main, eval_main, train_main


class TestCLIMainFunctions:
    """Test main CLI functions."""

    @patch('viberl.cli.SnakeGameEnv')
    @patch('viberl.cli.REINFORCEAgent')
    @patch('viberl.cli.train_agent')
    def test_train_main_basic(self, mock_train_agent, mock_agent_class, mock_env_class):  # noqa: ANN001
        """Test train_main function with basic arguments."""
        # Mock environment
        mock_env = Mock()
        mock_env_class.return_value = mock_env

        # Mock agent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.policy_network = Mock()

        # Mock training function
        mock_train_agent.return_value = [10.0, 12.0, 15.0]

        # Test with minimal arguments
        test_args = ['viberl-train', '--episodes', '10']

        with patch.object(sys, 'argv', test_args):
            train_main()

        # Verify environment was created
        mock_env_class.assert_called_once_with(grid_size=15)

        # Verify agent was created
        mock_agent_class.assert_called_once()

        # Verify training was called
        mock_train_agent.assert_called_once()

    @patch('viberl.cli.SnakeGameEnv')
    @patch('viberl.cli.REINFORCEAgent')
    @patch('viberl.cli.evaluate_agent')
    def test_eval_main_basic(self, mock_evaluate_agent, mock_agent_class, mock_env_class):  # noqa: ANN001
        """Test eval_main function with basic arguments."""
        # Mock environment
        mock_env = Mock()
        mock_env_class.return_value = mock_env

        # Mock agent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.policy_network = Mock()
        mock_agent.load_policy = Mock()

        # Mock evaluation function
        mock_evaluate_agent.return_value = [8.0, 12.0, 10.0]

        # Test with minimal arguments
        test_args = ['viberl-eval', '--model-path', 'test_model.pth']

        with patch.object(sys, 'argv', test_args):
            eval_main()

        # Verify environment was created with render mode
        mock_env_class.assert_called_once_with(render_mode=None, grid_size=15)

        # Verify agent was created and model was loaded
        mock_agent_class.assert_called_once()
        mock_agent.load_policy.assert_called_once_with('test_model.pth')

    @patch('viberl.cli.SnakeGameEnv')
    def test_demo_main_basic(self, mock_env_class):  # noqa: ANN001
        """Test demo_main function with basic arguments."""
        # Mock environment
        mock_env = Mock()
        mock_env_class.return_value = mock_env
        mock_env.reset.return_value = (None, {})
        mock_env.step.return_value = (None, 0, True, False, {'score': 5})
        mock_env.action_space.sample.return_value = 0

        # Test with minimal arguments
        test_args = ['viberl-demo', '--episodes', '2']

        with patch.object(sys, 'argv', test_args):
            demo_main()

        # Verify environment was created with render mode
        mock_env_class.assert_called_once_with(render_mode='human', grid_size=15)

        # Verify episodes were run
        assert mock_env.reset.call_count == 2
        assert mock_env.step.call_count >= 2
        assert mock_env.close.call_count == 1


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_train_main_help(self):
        """Test train_main help output."""
        test_args = ['tiny-rl-train', '--help']

        with patch.object(sys, 'argv', test_args):
            try:
                train_main()
            except SystemExit as e:
                # argparse calls sys.exit(0) after help
                assert e.code == 0

    def test_eval_main_model_path_required(self):
        """Test that eval_main requires model path."""
        test_args = ['tiny-rl-eval']

        with patch.object(sys, 'argv', test_args):
            try:
                eval_main()
                pytest.fail('Should have raised SystemExit due to missing model path')
            except SystemExit as e:
                # Should exit with error due to missing required argument
                assert e.code != 0

    def test_demo_main_default_episodes(self):
        """Test demo_main with default episode count."""
        test_args = ['viberl-demo']

        with (
            patch.object(sys, 'argv', test_args),
            patch('viberl.cli.SnakeGameEnv') as mock_env_class,
        ):
            mock_env = Mock()
            mock_env_class.return_value = mock_env
            mock_env.reset.return_value = (None, {})
            mock_env.step.return_value = (None, 0, True, False, {})
            mock_env.action_space.sample.return_value = 0

            demo_main()

            # Should use default of 5 episodes
            assert mock_env.reset.call_count == 5
