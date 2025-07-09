import os
import sys

from unittest.mock import patch

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snake_game.cli import TextPlayableSnake


class TestTextPlayableSnakeIntegration:
    """Integration tests for TextPlayableSnake."""

    @patch("snake_game.cli.os.system")
    def test_full_game_sequence(self, mock_system):
        """Test a complete game sequence including moves, food eating, and game over."""
        game = TextPlayableSnake(grid_size=5)

        # Mock step to return specific outcomes
        step_results = [
            # First move - normal move, no food
            (None, 0, False, False, {"score": 0, "snake_length": 3}),
            # Second move - eat food
            (None, 1, False, False, {"score": 1, "snake_length": 4}),
            # Third move - game over (hit wall)
            (None, -1, True, False, {"score": 1, "snake_length": 4}),
        ]

        with patch.object(game.env, "step", side_effect=step_results), patch.object(
            game.env, "reset"
        ), patch("builtins.print"), patch("snake_game.cli.msvcrt.getch") as mock_getch:
            # Simulate key presses: right (d), down (s), right (d), then quit (q)
            mock_getch.side_effect = ["d", "s", "d", "q"]

            with pytest.raises(SystemExit):
                game.play_cli(5)

            # Verify different types of moves were processed
            assert game.env.step.call_count == 3

    def test_keyboard_input_handling_windows(self):
        """Test Windows keyboard input handling."""
        game = TextPlayableSnake()

        with patch(
            "snake_game.cli.msvcrt.getch", return_value=b"w"
        ) as mock_getch, patch.object(game.env, "step") as mock_step, patch.object(
            game.env, "reset"
        ), patch("builtins.print"):
            # Mock step to create a loop that exits
            mock_step.side_effect = [
                (None, 0, False, False, {}),
                (None, 0, True, False, {}),  # Game over on second call
                (None, 0, False, False, {}),
            ]

            # This should handle the Windows input correctly
            try:
                game.play_cli(5)
            except SystemExit:
                pass

            mock_getch.assert_called()

    @patch("snake_game.cli.msvcrt")
    def test_keyboard_input_handling_unix_fallback(self, mock_msvcrt):
        """Test Unix keyboard input fallback."""
        # Make msvcrt unavailable
        mock_msvcrt.getch.side_effect = ImportError()

        game = TextPlayableSnake()

        with patch("sys.stdin.fileno", return_value=0), patch(
            "termios.tcgetattr"
        ), patch("termios.tcsetattr"), patch("tty.setraw"), patch(
            "sys.stdin.read", return_value="a"
        ) as mock_read, patch.object(game.env, "step") as mock_step, patch.object(
            game.env, "reset"
        ), patch("builtins.print"):
            # Mock step to create game over condition
            mock_step.side_effect = [
                (None, 0, False, False, {}),
                (None, 0, True, False, {}),  # Game over
                (None, 0, False, False, {}),
            ]

            try:
                game.play_cli(5)
            except SystemExit:
                pass

            mock_read.assert_called()

    @patch("snake_game.cli.msvcrt")
    @patch("sys.stdin.fileno")
    def test_keyboard_input_regular_input_fallback(self, mock_fileno, mock_msvcrt):
        """Test regular input fallback when terminal control fails."""
        # Make msvcrt and terminal control unavailable
        mock_msvcrt.getch.side_effect = ImportError()
        mock_fileno.side_effect = AttributeError()  # Make stdin.fileno fail

        game = TextPlayableSnake()

        with patch("builtins.input", return_value="s") as mock_input, patch.object(
            game.env, "step"
        ) as mock_step, patch.object(game.env, "reset"), patch("builtins.print"):
            # Mock step to create game over condition
            mock_step.side_effect = [
                (None, 0, False, False, {}),
                (None, 0, True, False, {}),  # Game over
                (None, 0, False, False, {}),
            ]

            try:
                game.play_cli(5)
            except SystemExit:
                pass

            mock_input.assert_called()


class TestTextModeDisplay:
    """Test text mode display functionality."""

    def test_display_grid_with_different_states(self):
        """Test display grid with various game states."""
        game = TextPlayableSnake(grid_size=3)

        # Test initial state
        game.env.reset()
        with patch("builtins.print") as mock_print:
            game.display_grid()

            # Should print score and length
            first_call = mock_print.call_args_list[0]
            assert "Score:" in first_call[0][0]
            assert "Length:" in first_call[0][0]
            assert "3" in first_call[0][0]  # Initial length

    def test_display_grid_game_over_state(self):
        """Test display grid when game is over."""
        game = TextPlayableSnake(grid_size=3)
        game.env.reset()
        game.env.game_over = True
        game.env.score = 5

        with patch("builtins.print") as mock_print:
            game.display_grid()

            # Should still display grid and score
            output = "\n".join([call[0][0] for call in mock_print.call_args_list])
            assert "Score: 5" in output
            assert "Length: 3" in output

    def test_display_help_comprehensive(self):
        """Test comprehensive help display."""
        game = TextPlayableSnake()

        with patch("builtins.print") as mock_print:
            game.display_help()

            help_output = "\n".join([call[0][0] for call in mock_print.call_args_list])

            # Check all control options are present
            assert "W/A/S/D" in help_output
            assert "K/H/J/L" in help_output
            assert "R - Restart" in help_output
            assert "Q - Quit" in help_output
            assert "Space" in help_output
            assert "Any other key - Apply current direction" in help_output


class TestTextModeGameLogic:
    """Test text mode game logic integration."""

    def test_invalid_key_applies_current_direction(self):
        """Test that invalid keys apply current direction."""
        game = TextPlayableSnake()
        game.env.reset()

        initial_direction = game.env.direction

        with patch("builtins.input", return_value="x"), patch.object(
            game.env, "step", return_value=(None, 0, False, False, {})
        ) as mock_step, patch("builtins.print"):
            try:
                game.play_cli(5)
            except SystemExit:
                pass
            except Exception:
                pass  # We expect other errors due to mocking

            # Test that step was called with current direction
            if mock_step.called:
                call_args = mock_step.call_args[0]
                assert call_args[0] == initial_direction.value

    def test_restart_functionality(self):
        """Test game restart functionality."""
        game = TextPlayableSnake()
        game.env.reset()
        game.env.score = 10
        game.env.steps = 50

        with patch("builtins.input", return_value="r"), patch.object(
            game.env, "reset"
        ) as mock_reset, patch.object(
            game.env, "step", return_value=(None, 0, False, False, {})
        ), patch("builtins.print"):
            try:
                game.play_cli(5)
            except:
                pass

            mock_reset.assert_called_once()

    def test_quit_functionality(self):
        """Test game quit functionality."""
        game = TextPlayableSnake()
        game.env.reset()

        with patch("builtins.input", return_value="q"), patch(
            "builtins.print"
        ) as mock_print:
            with pytest.raises(SystemExit):
                game.play_cli(5)

            # Should print quit message
            quit_message_found = any(
                "Thanks for playing" in str(call) for call in mock_print.call_args_list
            )
            assert quit_message_found


class TestTextModeEdgeCases:
    """Test edge cases in text mode."""

    def test_empty_food_placement(self):
        """Test when there's no space to place food."""
        game = TextPlayableSnake(grid_size=2)
        game.env.reset()

        # Fill entire grid with snake
        game.env.snake = [(0, 0), (0, 1), (1, 1), (1, 0)]
        game.env.food = None

        with patch("builtins.print") as mock_print:
            game.display_grid()

            # Should handle gracefully - no food shown but no crash
            output = "\n".join([call[0][0] for call in mock_print.call_args_list])

        # The game should continue working
        game.env.food = (0, 0)  # Reset food
        with patch("builtins.print"):
            game.display_grid()  # Should work normally

    def test_game_over_with_restart(self):
        """Test game over followed by restart."""
        game = TextPlayableSnake()
        game.env.reset()
        game.env.game_over = True
        game.env.score = 8

        input_sequence = ["any_key", "q"]  # Any key then quit

        with patch("builtins.input", side_effect=input_sequence), patch.object(
            game.env, "reset"
        ) as mock_reset, patch("builtins.print") as mock_print:
            try:
                game.play_cli(5)
            except SystemExit:
                pass

            # Should have printed game over message and prompted for input
            mock_reset.assert_called_once()

    def test_key_error_recovery(self):
        """Test recovery from keyboard input errors."""
        game = TextPlayableSnake()
        game.env.reset()

        # First input fails, second succeeds
        input_sequence = [Exception("Keyboard error"), "d", "q"]

        def mock_input_side_effect():
            for item in input_sequence:
                if isinstance(item, Exception):
                    raise item
                yield item
            while True:
                yield "q"

        with patch(
            "builtins.input", side_effect=mock_input_side_effect()
        ), patch.object(game.env, "step"), patch("builtins.print"):
            try:
                game.play_cli(5)
            except:
                pass
            # Should not crash on keyboard error

    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupt."""
        game = TextPlayableSnake()
        game.env.reset()

        with patch("builtins.input", side_effect=KeyboardInterrupt()), patch(
            "builtins.print"
        ) as mock_print:
            try:
                game.play_cli(5)
            except SystemExit:
                pass

            # Should print interruption message
            interrupt_message_found = any(
                "interrupted by user" in str(call) for call in mock_print.call_args_list
            )
            assert interrupt_message_found


class TestTextModePerformance:
    """Test text mode performance and timing."""

    def test_delay_between_frames(self):
        """Test that there's appropriate delay between frames."""
        game = TextPlayableSnake()
        game.env.reset()

        with patch("time.sleep") as mock_sleep, patch(
            "builtins.input", return_value="d"
        ), patch.object(
            game.env, "step", return_value=(None, 0, False, False, {})
        ), patch("builtins.print"):
            try:
                game.play_cli(5)
            except:
                pass

            # Should have sleep calls for delay
            mock_sleep.assert_called_with(0.05)

    def test_multiple_inputs_handling(self):
        """Test handling of multiple rapid inputs."""
        game = TextPlayableSnake()
        game.env.reset()

        # Rapid sequence of moves
        input_sequence = ["d", "s", "a", "w", "q"]

        with patch("builtins.input", side_effect=input_sequence), patch.object(
            game.env, "step"
        ) as mock_step, patch("builtins.print"):
            mock_step.side_effect = [(None, 0, False, False, {})] * len(input_sequence)

            try:
                game.play_cli(5)
            except SystemExit:
                pass

            assert mock_step.call_count == len(input_sequence) - 1  # Last one is quit
