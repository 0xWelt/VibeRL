from __future__ import annotations

import argparse
import os
import sys
import time


# Import raw_input methods at top level
try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

from .core import Direction, SnakeGameEnv


# Try to import pygame for GUI mode, but allow CLI mode to work without it
try:
    import pygame
except ImportError:
    pygame = None


def clear_screen_safe() -> None:
    """Clear terminal screen safely."""
    command = 'cls' if os.name == 'nt' else 'clear'
    try:
        # Use subprocess for security over os.system
        import subprocess

        subprocess.run([command], shell=True, check=False)  # noqa: S602
    except (OSError, subprocess.SubprocessError):
        # Fallback to printing newlines if terminal commands fail
        print('\n' * 50)


def get_input_safe() -> str:
    """Get single keypress input safely on all platforms."""
    try:
        if os.name == 'nt' and msvcrt is not None:
            try:
                return msvcrt.getch().decode('utf-8').lower()
            except UnicodeDecodeError:
                return input().lower()
        elif termios is not None and tty is not None:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                return sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (AttributeError, termios.error, OSError):
        return input().lower()


"""
Command-line interface for Snake Game.
"""


class HumanPlayableSnake:
    """Wrapper for human gameplay with keyboard controls."""

    def __init__(self, grid_size: int = 20):
        self.env = SnakeGameEnv(render_mode='human', grid_size=grid_size)
        self.env.reset()
        self.action_map = {
            pygame.K_UP: Direction.UP.value,
            pygame.K_RIGHT: Direction.RIGHT.value,
            pygame.K_DOWN: Direction.DOWN.value,
            pygame.K_LEFT: Direction.LEFT.value,
        }

    def play(self) -> None:
        """Start human-playable game."""
        print('Snake Game - Control with arrow keys, R to restart, Q to quit')

        running = True

        while running:
            action = None

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        action = None
                    elif event.key in self.action_map:
                        action = self.action_map[event.key]

            if action is not None:
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    print(f'Game Over! Final Score: {info["score"]}')

            self.env.render()

        self.env.close()


def human_mode(args: argparse.Namespace | None = None) -> None:
    """Run game in human-playable mode."""
    parser = argparse.ArgumentParser(description='Play Snake game with keyboard controls')
    parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )
    parsed_args = parser.parse_args() if args is None else args

    game = HumanPlayableSnake(grid_size=parsed_args.grid_size)
    game.play()


def ai_mode(args: argparse.Namespace | None = None) -> None:
    """Run game in AI mode with random actions for demonstration."""
    parser = argparse.ArgumentParser(description='Run Snake environment with AI')
    parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )
    parser.add_argument(
        '--episodes', type=int, default=1, help='Number of episodes to run (default: 1)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        default=True,
        help='Enable rendering (default: True)',
    )
    parsed_args = parser.parse_args() if args is None else args

    print('🐍 Snake Gymnasium - AI Demo Mode')
    print(
        'Use with your reinforcement learning algorithms by importing: from snake_game import SnakeGameEnv'
    )

    env = SnakeGameEnv(
        render_mode='human' if parsed_args.render else None,
        grid_size=parsed_args.grid_size,
    )

    try:
        for episode in range(parsed_args.episodes):
            obs, info = env.reset()
            total_reward = 0

            print(f'\nEpisode {episode + 1}/{parsed_args.episodes}')

            while True:
                action = env.action_space.sample()  # Random action for demo
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if parsed_args.render:
                    env.render()

                if terminated or truncated:
                    print(f'Episode finished! Score: {info["score"]}, Total reward: {total_reward}')
                    break

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        env.close()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🐍 Snake Game with Gymnasium Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play the game yourself
  snake-human --grid-size 20

  # Watch AI play randomly
  snake-ai --episodes 5

  # AI without rendering
  snake-ai --episodes 10 --render

Import in your code:
  from snake_game import SnakeGameEnv, Direction
  env = SnakeGameEnv(render_mode="human")
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Human command
    human_parser = subparsers.add_parser('human', help='Play the game yourself')
    human_parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )

    # AI command
    ai_parser = subparsers.add_parser('ai', help='Run AI mode')
    ai_parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )
    ai_parser.add_argument(
        '--episodes', type=int, default=1, help='Number of episodes to run (default: 1)'
    )
    ai_parser.add_argument(
        '--no-render',
        action='store_false',
        dest='render',
        help='Disable rendering for faster execution',
    )

    # Quick command aliases
    subparsers.add_parser('play', help='Alias for human command')
    subparsers.add_parser('demo', help='Alias for ai command')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'human':
        human_mode(args)
    elif args.command == 'ai':
        ai_mode(args)
    elif args.command == 'play':
        args_copy = argparse.Namespace(grid_size=20)
        human_mode(args_copy)
    elif args.command == 'demo':
        args_copy = argparse.Namespace(grid_size=20, episodes=3, render=True)
        ai_mode(args_copy)


class TextPlayableSnake:
    """Text-based Snake game for terminal play (no GUI required)."""

    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        self.env = SnakeGameEnv(grid_size=grid_size)
        self.env.reset()

        # Keyboard mappings
        self.key_map = {
            'w': Direction.UP.value,
            'a': Direction.LEFT.value,
            's': Direction.DOWN.value,
            'd': Direction.RIGHT.value,
            ' ': Direction.UP.value,  # Space as up
            'k': Direction.UP.value,
            'h': Direction.LEFT.value,
            'j': Direction.DOWN.value,
            'l': Direction.RIGHT.value,
        }

    def clear_screen(self):
        """Clear terminal screen safely."""
        clear_screen_safe()

    def display_grid(self) -> None:
        """Display the current game state in text format."""
        print(f'Score: {self.env.score} | Length: {len(self.env.snake)}')
        print('┌' + '─' * self.grid_size + '┐')

        for i in range(self.grid_size):
            print('│', end='')
            for j in range(self.grid_size):
                if (i, j) in self.env.snake[:-1]:
                    print('█', end='')  # Snake body
                elif (i, j) == self.env.snake[-1]:
                    print('☻', end='')  # Snake head
                elif self.env.food and (i, j) == self.env.food:
                    print('●', end='')  # Food
                else:
                    print(' ', end='')  # Empty space
            print('│')

        print('└' + '─' * self.grid_size + '┘')

    def display_help(self) -> None:
        """Display help information."""
        print('\nControls:')
        print('  W/A/S/D or K/H/J/L - Move snake')
        print('  R - Restart game')
        print('  Q - Quit game')
        print('  Space - Move up')
        print('  Any other key - Apply current direction')
        print()

    def play_cli(self, grid_size: int) -> None:
        """Play Snake game in CLI mode."""
        print('🐍 CLI Snake Game - No GUI needed!')
        print('Use W/A/S/D or K/H/J/L to control the snake')
        print('R to restart, Q to quit\n')

        # Initialize environment
        self.env.reset()

        running = True

        try:
            while running:
                # Clear screen and display current state
                self.clear_screen()
                print('🐍 CLI Snake Game - Use W/A/S/D to move, R to restart, Q to quit\n')

                self.display_grid()
                self.display_help()

                if self.env.game_over:
                    print(f'💀 GAME OVER! Final Score: {self.env.score}')
                    print(f'Final Snake Length: {len(self.env.snake)}')
                    print('\nPress any key to start a new game, or Q to quit')

                    try:
                        choice = input().lower().strip()
                        if choice == 'q':
                            running = False
                        else:
                            self.env.reset()
                            print('New game started!')
                            time.sleep(1)
                        continue
                    except KeyboardInterrupt:
                        print('\nGame interrupted.')
                        break

                # Get user input
                print('Enter move (WASD or HJKL): ', end='', flush=True)
                key = get_input_safe()

                action = None

                # Process input
                if key == 'q':
                    running = False
                    print('\nThanks for playing!')
                    break
                if key == 'r':
                    self.env.reset()
                    print('Game restarted!')
                elif key in self.key_map:
                    action = self.key_map[key]
                else:
                    # Apply current direction for any other key
                    direction = self.env.direction
                    action = direction.value

                if action is not None and not self.env.game_over:
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if terminated or truncated:
                        # Don't break, game over will be handled in next iteration
                        pass

                # Small delay for better gameplay experience
                time.sleep(0.05)

        except KeyboardInterrupt:
            print('\n\nGame interrupted by user. Thanks for playing!')
        except Exception as e:  # noqa: BLE001
            print(f'Error occurred: {e}')
        finally:
            print('\nGame ended.')


def text_mode(args: argparse.Namespace | None = None) -> None:
    """Run game in text-based CLI mode."""
    parser = argparse.ArgumentParser(description='Play Snake game in text-based CLI mode')
    parser.add_argument(
        '--grid-size',
        type=int,
        default=15,
        help='Grid size for the game (default: 15, max: 25)',
    )
    parsed_args = parser.parse_args() if args is None else args

    # Validate grid size
    if parsed_args.grid_size < 5:
        parsed_args.grid_size = 5
        print('⚠️  Grid size must be at least 5. Using 5.')
    elif parsed_args.grid_size > 25:
        parsed_args.grid_size = 25
        print('⚠️  Grid size too large. Using 25.')

    game = TextPlayableSnake(grid_size=parsed_args.grid_size)
    game.play_cli(parsed_args.grid_size)


if __name__ == '__main__':
    main()


# Update the main function to include text mode


def main() -> None:
    """Main CLI entry point with text mode support."""
    parser = argparse.ArgumentParser(
        description='🐍 Snake Game with Gymnasium Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play the game yourself (with GUI)
  snake-game human --grid-size 20

  # Play the game yourself (text-based, no GUI required)
  snake-game text --grid-size 15

  # Watch AI play randomly
  snake-ai --episodes 5

  # AI without rendering
  snake-ai --episodes 10 --no-render

Import in your code:
  from snake_game import SnakeGameEnv, Direction
  env = SnakeGameEnv(render_mode="human")
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Human command (GUI mode)
    human_parser = subparsers.add_parser('human', help='Play the game with GUI')
    human_parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )

    # Text command (CLI mode, no GUI)
    text_parser = subparsers.add_parser(
        'text', help='Play the game in text-based CLI mode (no GUI required)'
    )
    text_parser.add_argument(
        '--grid-size',
        type=int,
        default=15,
        help='Grid size for the game (default: 15, max: 25)',
    )

    # AI command
    ai_parser = subparsers.add_parser('ai', help='Run AI mode')
    ai_parser.add_argument(
        '--grid-size', type=int, default=20, help='Grid size for the game (default: 20)'
    )
    ai_parser.add_argument(
        '--episodes', type=int, default=1, help='Number of episodes to run (default: 1)'
    )
    ai_parser.add_argument(
        '--render',
        action='store_true',
        default=True,
        help='Enable rendering (default: True)',
    )

    # Quick command aliases
    subparsers.add_parser('play', help='Alias for human command')
    subparsers.add_parser('demo', help='Alias for ai command')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'human':
        human_mode(args)
    elif args.command == 'text':
        text_mode(args)
    elif args.command == 'ai':
        ai_mode(args)
    elif args.command == 'play':
        args_copy = argparse.Namespace(grid_size=20)
        human_mode(args_copy)
    elif args.command == 'demo':
        args_copy = argparse.Namespace(grid_size=20, episodes=3, render=True)
        ai_mode(args_copy)
