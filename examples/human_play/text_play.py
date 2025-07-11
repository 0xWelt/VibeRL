#!/usr/bin/env python3
"""
Text-based Snake Game for environments without display support

This script allows you to play the Snake game in terminal/console mode
without requiring pygame display support.
"""

import sys
from viberl.envs import SnakeGameEnv


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_game_state(env):
    """Print the game state as text."""
    obs = env._get_observation()
    grid_size = env.grid_size
    
    print("┌" + "─" * (grid_size * 2 + 1) + "┐")
    for y in range(grid_size):
        print("│", end=" ")
        for x in range(grid_size):
            cell = obs[y, x]
            if cell == 0:  # Empty
                print("·", end=" ")
            elif cell == 1:  # Snake body
                print("●", end=" ")
            elif cell == 2:  # Snake head
                print("◆", end=" ")
            elif cell == 3:  # Food
                print("★", end=" ")
        print("│")
    print("└" + "─" * (grid_size * 2 + 1) + "┘")
    print(f"Score: {env.score}")


def play_text_game(grid_size=15):
    """Play Snake game in text mode."""
    env = SnakeGameEnv(grid_size=grid_size)
    obs, info = env.reset()
    
    print("🐍 Text Snake Game")
    print("=================")
    print("Controls:")
    print("  W/S/A/D: Move Up/Down/Left/Right")
    print("  Q: Quit")
    print("  R: Restart")
    print()
    
    score = 0
    
    # Map keys to actions
    key_map = {
        'w': 0,  # UP
        's': 2,  # DOWN
        'a': 3,  # LEFT
        'd': 1,  # RIGHT
        'W': 0, 'S': 2, 'A': 3, 'D': 1
    }
    
    try:
        while True:
            clear_screen()
            print_game_state(env)
            
            # Get user input
            print("\nEnter move (W/S/A/D): ", end="", flush=True)
            key = sys.stdin.read(1)
            
            if key.lower() == 'q':
                print("\nGame ended by user")
                break
            elif key.lower() == 'r':
                obs, info = env.reset()
                score = 0
                print("Game restarted!")
                continue
            elif key in key_map:
                action = key_map[key]
                obs, reward, terminated, truncated, info = env.step(action)
                score = info['score']
                
                if terminated or truncated:
                    clear_screen()
                    print_game_state(env)
                    print(f"\n💀 Game Over! Final Score: {score}")
                    print("Press R to restart or Q to quit")
                    
                    next_action = sys.stdin.read(1)
                    if next_action.lower() == 'r':
                        obs, info = env.reset()
                        score = 0
                        continue
                    else:
                        break
            else:
                print(f"Invalid key: {key}")
                
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Snake game in terminal')
    parser.add_argument('--grid-size', type=int, default=15,
                       help='Grid size for the game (default: 15)')
    
    args = parser.parse_args()
    
    try:
        play_text_game(args.grid_size)
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)