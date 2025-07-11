#!/usr/bin/env python3
"""
Simple human-playable Snake Game using VibeRL
Run this in a Python environment with display support
"""

import pygame
from viberl.envs import SnakeGameEnv, Direction


def play_snake_human(grid_size=15):
    """Simple snake game for human players."""
    pygame.init()
    
    # Create environment
    env = SnakeGameEnv(render_mode='human', grid_size=grid_size)
    obs, info = env.reset()
    
    print("🐍 Snake Game - Human Controls")
    print("=============================")
    print("Use arrow keys to control the snake")
    print("Press Q to quit")
    print(f"Starting game with {grid_size}x{grid_size} grid...")
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Handle events
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_UP:
                        action = Direction.UP.value
                    elif event.key == pygame.K_RIGHT:
                        action = Direction.RIGHT.value
                    elif event.key == pygame.K_DOWN:
                        action = Direction.DOWN.value
                    elif event.key == pygame.K_LEFT:
                        action = Direction.LEFT.value
            
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                score = info['score']
                
                if terminated or truncated:
                    print(f"Game Over! Final Score: {score}")
                    obs, info = env.reset()  # Auto-restart
                    
            # Limit to 10 FPS for smooth play
            clock.tick(10)
            
    except KeyboardInterrupt:
        print("\nGame interrupted")
    finally:
        env.close()
        pygame.quit()
        print("Game ended")


if __name__ == "__main__":
    play_snake_human(grid_size=15)