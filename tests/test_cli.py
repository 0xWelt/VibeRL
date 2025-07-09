import pytest
import argparse
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_game.cli import (
    HumanPlayableSnake, 
    TextPlayableSnake,
    human_mode, 
    ai_mode, 
    text_mode, 
    main
)
from snake_game.core import SnakeGameEnv, Direction


class TestHumanPlayableSnake:
    """Test HumanPlayableSnake class."""
    
    def test_initialization(self):
        with patch('snake_game.cli.pygame'):
            game = HumanPlayableSnake(grid_size=25)
            assert game.grid_size == 25
            assert isinstance(game.env, SnakeGameEnv)
            assert game.action_map[pygame.K_UP] == Direction.UP.value
            assert game.action_map[pygame.K_RIGHT] == Direction.RIGHT.value
            assert game.action_map[pygame.K_DOWN] == Direction.DOWN.value
            assert game.action_map[pygame.K_LEFT] == Direction.LEFT.value
    
    @patch('snake_game.cli.pygame')
    def test_play_quit_event(self, mock_pygame):
        # Mock pygame events to simulate quit
        mock_pygame.QUIT = pygame.QUIT
        mock_pygame.event.get.return_value = [Mock(type=pygame.QUIT)]
        
        game = HumanPlayableSnake()
        game.play()
        
        # Should exit gracefully
        mock_pygame.event.get.assert_called()
    
    @patch('snake_game.cli.pygame')
    def test_play_key_events(self, mock_pygame):
        # Mock pygame constants
        mock_pygame.QUIT = pygame.QUIT
        mock_pygame.KEYDOWN = pygame.KEYDOWN
        mock_pygame.K_q = pygame.K_q
        mock_pygame.K_r = pygame.K_r
        mock_pygame.K_UP = pygame.K_UP
        
        # Mock events: first quit key, then restart, then arrow key
        mock_pygame.event.get.side_effect = [
            [Mock(type=pygame.KEYDOWN, key=pygame.K_q)],  # Quit first
            [Mock(type=pygame.KEYDOWN, key=pygame.K_r)],  # Then restart
            [Mock(type=pygame.KEYDOWN, key=pygame.K_UP)],  # Then move
            [Mock(type=pygame.QUIT)]  # Finally quit
        ]
        
        game = HumanPlayableSnake()
        with patch.object(game.env, 'step') as mock_step, \
             patch.object(game.env, 'render'):
            mock_step.return_value = (None, 0, False, False, {})
            game.play()
        
        mock_step.assert_called_with(Direction.UP.value)


class TestTextPlayableSnake:
    """Test TextPlayableSnake class."""
    
    def test_initialization(self):
        game = TextPlayableSnake(grid_size=15)
        assert game.grid_size == 15
        assert isinstance(game.env, SnakeGameEnv)
        
        # Check key mappings
        assert game.key_map['w'] == Direction.UP.value
        assert game.key_map['a'] == Direction.LEFT.value
        assert game.key_map['s'] == Direction.DOWN.value
        assert game.key_map['d'] == Direction.RIGHT.value
        assert game.key_map[' '] == Direction.UP.value
        assert game.key_map['k'] == Direction.UP.value
        assert game.key_map['h'] == Direction.LEFT.value
        assert game.key_map['j'] == Direction.DOWN.value
        assert game.key_map['l'] == Direction.RIGHT.value
    
    @patch('snake_game.cli.os.system')
    def test_clear_screen(self, mock_system):
        game = TextPlayableSnake()
        game.clear_screen()
        mock_system.assert_called_with('cls' if os.name == 'nt' else 'clear')
    
    def test_display_grid_basic(self):
        game = TextPlayableSnake(grid_size=5)
        game.env.reset()
        
        with patch('builtins.print') as mock_print:
            game.display_grid()
            
            # Check that print was called multiple times
            assert mock_print.call_count > 5
            
            # Check score display
            score_call = mock_print.call_args_list[0]
            assert "Score:" in score_call[0][0]
            assert "Length:" in score_call[0][0]
            
            # Check grid borders
            border_calls = [call[0][0] for call in mock_print.call_args_list if call[0][0].startswith('┌') or call[0][0].startswith('└')]
            assert len(border_calls) >= 2
    
    def test_display_help(self):
        game = TextPlayableSnake()
        
        with patch('builtins.print') as mock_print:
            game.display_help()
            
            # Check help content
            help_text = '\n'.join([call[0][0] for call in mock_print.call_args_list])
            assert "Controls:" in help_text
            assert "W/A/S/D" in help_text
            assert "K/H/J/L" in help_text
            assert "R - Restart" in help_text
            assert "Q - Quit" in help_text


class TestCLIModes:
    """Test CLI mode functions."""
    
    @patch('snake_game.cli.HumanPlayableSnake')
    def test_human_mode(self, mock_game_class):
        mock_game = Mock()
        mock_game_class.return_value = mock_game
        
        args = argparse.Namespace(grid_size=30)
        human_mode(args)
        
        mock_game_class.assert_called_once_with(grid_size=30)
        mock_game.play.assert_called_once()
    
    @patch('snake_game.cli.SnakeGameEnv')
    def test_ai_mode_single_episode(self, mock_env_class):
        mock_env = Mock()
        mock_env_class.return_value = mock_env
        mock_env.reset.return_value = (None, {})
        mock_env.step.return_value = (None, 0, True, False, {"score": 5})
        mock_env.action_space.sample.return_value = 0
        
        args = argparse.Namespace(grid_size=25, episodes=1, render=True)
        
        with patch('builtins.print'):
            ai_mode(args)
        
        mock_env_class.assert_called_once_with(render_mode="human", grid_size=25)
        mock_env.reset.assert_called_once()
        mock_env.step.assert_called()
        mock_env.close.assert_called_once()
    
    @patch('snake_game.cli.SnakeGameEnv')
    def test_ai_mode_multiple_episodes(self, mock_env_class):
        mock_env = Mock()
        mock_env_class.return_value = mock_env
        mock_env.reset.return_value = (None, {})
        mock_env.step.return_value = (None, 0, True, False, {"score": 0})
        mock_env.action_space.sample.return_value = 0
        
        args = argparse.Namespace(grid_size=20, episodes=3, render=False)
        
        with patch('builtins.print'):
            ai_mode(args)
        
        # Should reset twice (once per episode after first)
        assert mock_env.reset.call_count == 3
        mock_env.close.assert_called_once()
    
    @patch('snake_game.cli.TextPlayableSnake')
    def test_text_mode_valid_grid_size(self, mock_game_class):
        mock_game = Mock()
        mock_game_class.return_value = mock_game
        
        args = argparse.Namespace(grid_size=15)
        text_mode(args)
        
        mock_game_class.assert_called_once_with(grid_size=15)
        mock_game.play_cli.assert_called_once_with(15)
    
    @patch('snake_game.cli.TextPlayableSnake')
    def test_text_mode_grid_size_bounds(self, mock_game_class):
        mock_game = Mock()
        mock_game_class.return_value = mock_game
        
        # Test minimum bound
        args = argparse.Namespace(grid_size=3)
        with patch('builtins.print') as mock_print:
            text_mode(args)
            
            # Should print warning about minimum size
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'must be at least 5' in str(call)]
            assert len(warning_calls) > 0
        
        mock_game_class.assert_called_with(grid_size=5)  # Should be clamped to 5
        
        # Test maximum bound
        args = argparse.Namespace(grid_size=30)
        with patch('builtins.print') as mock_print:
            text_mode(args)
            
            # Should print warning about maximum size
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'too large' in str(call)]
            assert len(warning_calls) > 0
        
        mock_game_class.assert_called_with(grid_size=25)  # Should be clamped to 25


class TestMainCLI:
    """Test main CLI functionality."""
    
    @patch('snake_game.cli.human_mode')
    def test_main_human_command(self, mock_human_mode):
        test_args = ["snake-game", "human", "--grid-size", "25"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_human_mode.assert_called_once()
        args = mock_human_mode.call_args[0][0]
        assert args.grid_size == 25
    
    @patch('snake_game.cli.text_mode')
    def test_main_text_command(self, mock_text_mode):
        test_args = ["snake-game", "text", "--grid-size", "15"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_text_mode.assert_called_once()
        args = mock_text_mode.call_args[0][0]
        assert args.grid_size == 15
    
    @patch('snake_game.cli.ai_mode')
    def test_main_ai_command(self, mock_ai_mode):
        test_args = ["snake-game", "ai", "--episodes", "5", "--grid-size", "30"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_ai_mode.assert_called_once()
        args = mock_ai_mode.call_args[0][0]
        assert args.episodes == 5
        assert args.grid_size == 30
        assert args.render is True
    
    @patch('snake_game.cli.ai_mode')
    def test_main_ai_command_no_render(self, mock_ai_mode):
        test_args = ["snake-game", "ai", "--no-render"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        args = mock_ai_mode.call_args[0][0]
        assert args.render is False
    
    @patch('snake_game.cli.human_mode')
    def test_main_play_alias(self, mock_human_mode):
        test_args = ["snake-game", "play", "--grid-size", "20"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_human_mode.assert_called_once()
    
    @patch('snake_game.cli.ai_mode')
    def test_main_demo_alias(self, mock_ai_mode):
        test_args = ["snake-game", "demo"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Demo should call ai_mode with default args
        mock_ai_mode.assert_called_once()
        args = mock_ai_mode.call_args[0][0]
        assert args.grid_size == 20
        assert args.episodes == 3
        assert args.render is True
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_no_command_shows_help(self, mock_stdout):
        test_args = ["snake-game"]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        output = mock_stdout.getvalue()
        assert "available commands" in output.lower() or "help" in output.lower()


class TestCommandLineIntegration:
    """Integration tests for command line interface."""
    
    def test_cli_help_output(self):
        """Test that help output contains expected information."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            try:
                test_args = ["snake-game", "--help"]
                with patch.object(sys, 'argv', test_args):
                    main()
            except SystemExit:
                pass  # argparse calls sys.exit after help
        
        output = mock_stdout.getvalue()
        assert "human" in output
        assert "text" in output
        assert "ai" in output
        assert "play" in output
        assert "demo" in output
    
    @patch('snake_game.cli.HumanPlayableSnake')
    def test_human_mode_with_default_args(self, mock_game_class):
        """Test human mode with no arguments."""
        args = None
        human_mode(args)
        
        # Should use default grid size of 20
        mock_game_class.assert_called_once_with(grid_size=20)
    
    @patch('snake_game.cli.SnakeGameEnv')
    def test_ai_mode_with_default_args(self, mock_env_class):
        """Test AI mode with no arguments."""
        args = None
        ai_mode(args)
        
        # Should use default values
        mock_env_class.assert_called_once_with(render_mode="human", grid_size=20)


class TestErrorHandling:
    """Test error handling in CLI components."""
    
    @patch('snake_game.cli.SnakeGameEnv')
    def test_ai_mode_keyboard_interrupt(self, mock_env_class):
        """Test handling of keyboard interrupt in AI mode."""
        mock_env = Mock()
        mock_env_class.return_value = mock_env
        mock_env.reset.return_value = (None, {})
        mock_env.step.return_value = (None, 0, False, False, {})
        mock_env.action_space.sample.return_value = 0
        
        # Simulate keyboard interrupt during episode
        mock_env.step.side_effect = [KeyboardInterrupt(), (None, 0, True, False, {})]
        
        args = argparse.Namespace(grid_size=20, episodes=1, render=False)
        
        with patch('builtins.print'):
            ai_mode(args)  # Should handle interrupt gracefully
        
        mock_env.close.assert_called_once()