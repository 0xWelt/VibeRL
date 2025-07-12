"""
Human-playable Snake Game using VibeRL

This script allows you to play the Snake game with keyboard controls.
"""

import sys

import pygame

from viberl.envs import Direction, SnakeGameEnv


def play_human_game(grid_size: int = 15) -> None:
    """Play Snake game with human controls."""

    # Initialize pygame display
    try:
        pygame.init()
        pygame.display.init()
    except pygame.error as e:
        print(f'❌ Display not available: {e}')
        print('💡 This script requires a display server (X11/Windows/macOS)')
        print('💡 Solutions:')
        print('   1. Run locally on your machine (not SSH/remote)')
        print('   2. Use X11 forwarding: ssh -X user@host')
        print('   3. Use VNC or similar remote desktop')
        return

    print('🎮 Initializing Snake game...')

    # Create environment with human rendering
    env = SnakeGameEnv(render_mode='human', grid_size=grid_size)
    print('✅ Environment created successfully')

    print('🎮 Snake Game - Human Mode')
    print('========================')
    print('Controls:')
    print('  Arrow Keys: Move snake')
    print('  R: Restart game')
    print('  Q: Quit game')
    print('  ESC: Exit')
    print()

    # Reset environment
    observation, info = env.reset()
    env.render()
    score = 0

    print('🎯 Game started! Use arrow keys to move...')

    # Game loop
    running = True
    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = env.reset()
                        env.render()
                        score = 0
                        print('🔄 Game restarted!')
                    elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        # Map arrow keys to actions
                        key_map = {
                            pygame.K_UP: Direction.UP.value,
                            pygame.K_DOWN: Direction.DOWN.value,
                            pygame.K_LEFT: Direction.LEFT.value,
                            pygame.K_RIGHT: Direction.RIGHT.value,
                        }
                        action = key_map[event.key]

                        # Take action
                        observation, reward, terminated, truncated, info = env.step(action)
                        env.render()
                        score = info['score']

                        if terminated or truncated:
                            print(f'💀 Game Over! Final Score: {score}')
                            print('🔄 Press R to restart or Q to quit')
                        else:
                            print(f'📊 Score: {score}')

            # Control frame rate
            pygame.time.delay(100)  # ~10 FPS for better responsiveness

    except KeyboardInterrupt:
        print('\n🛑 Game interrupted by user')
    finally:
        env.close()
        pygame.quit()
        print(f'\n🏁 Game ended. Final Score: {score}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Play Snake game as human')
    parser.add_argument(
        '--grid-size', type=int, default=15, help='Grid size for the game (default: 15)'
    )
    parser.add_argument(
        '--test', action='store_true', help='Quick test mode (skip actual gameplay)'
    )

    args = parser.parse_args()

    if args.test:
        print('🧪 Running in test mode...')
        try:
            pygame.init()
            env = SnakeGameEnv(render_mode='human', grid_size=args.grid_size)
            obs, info = env.reset()
            env.render()
            print('✅ Game initialized successfully!')
            print(f'Grid size: {args.grid_size}x{args.grid_size}')
            print('🎯 Game is ready for human play')
            env.close()
            pygame.quit()
        except pygame.error as e:
            print(f'❌ Error: {e}')
    else:
        try:
            play_human_game(args.grid_size)
        except KeyboardInterrupt:
            print('\nGame interrupted by user')
            sys.exit(0)
