#!/usr/bin/env python3
"""
Human-playable Snake Game using VibeRL

This script allows you to play the Snake game with keyboard controls.
"""

import pygame
import sys
from viberl.envs import SnakeGameEnv, Direction


def play_human_game(grid_size=15):
    """Play Snake game with human controls."""
    
    # Initialize pygame display
    pygame.init()
    
    # Create environment with human rendering
    env = SnakeGameEnv(render_mode='human', grid_size=grid_size)
    
    print("🎮 Snake Game - Human Mode")
    print("========================")
    print("Controls:")
    print("  Arrow Keys: Move snake")
    print("  R: Restart game")
    print("  Q: Quit game")
    print("  ESC: Exit")
    print()
    
    # Reset environment
    observation, info = env.reset()
    score = 0
    
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
                        score = 0
                        print("Game restarted!")
                    elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        # Map arrow keys to actions
                        key_map = {
                            pygame.K_UP: Direction.UP.value,
                            pygame.K_DOWN: Direction.DOWN.value,
                            pygame.K_LEFT: Direction.LEFT.value,
                            pygame.K_RIGHT: Direction.RIGHT.value
                        }
                        action = key_map[event.key]
                        
                        # Take action
                        observation, reward, terminated, truncated, info = env.step(action)
                        score = info['score']
                        
                        if terminated or truncated:
                            print(f"Game Over! Final Score: {score}")
                            print("Press R to restart or Q to quit")
                        else:
                            print(f"Score: {score}")
            
            # Control frame rate
            pygame.time.delay(150)  # ~6-7 FPS for human playability
            
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env.close()
        pygame.quit()
        print(f"\nGame ended. Final Score: {score}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Snake game as human')
    parser.add_argument('--grid-size', type=int, default=15, 
                       help='Grid size for the game (default: 15)')
    
    args = parser.parse_args()
    
    try:
        play_human_game(args.grid_size)
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)