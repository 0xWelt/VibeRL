#!/usr/bin/env python3
"""
Auto-Step Text Snake Game

This script runs the game with automatic movement every 0.3 seconds.
You can change direction with WASD keys (direct input, no enter needed) while the game auto-runs.
"""

import sys
import time
import threading
import tty
import termios
from viberl.envs import SnakeGameEnv


def getch():
    """Get single character from terminal without pressing enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")
    sys.stdout.flush()


def print_game_state(env, last_action):
    """Print the game state as text."""
    obs = env._get_observation()
    grid_size = env.grid_size
    
    # Direction symbols
    direction_symbols = ['↑', '→', '↓', '←']
    
    print("🐍 Auto-Step Snake Game")
    print("=" * 25)
    print(f"Score: {env.score} | Direction: {direction_symbols[last_action]}")
    print()
    
    # Print grid with consistent alignment
    border = "┌" + "─" * (grid_size * 2 + 1) + "┐"
    print(border)
    
    for y in range(grid_size):
        line = "│ "
        for x in range(grid_size):
            cell = obs[y, x]
            if cell == 0:  # Empty
                line += "· "
            elif cell == 1:  # Snake body
                line += "● "
            elif cell == 2:  # Snake head
                line += "◆ "
            elif cell == 3:  # Food
                line += "★ "
        line += "│"
        print(line)
    
    print("└" + "─" * (grid_size * 2 + 1) + "┘")
    
    print("\nControls:")
    print("  WASD: Change direction (direct input)")
    print("  Q: Quit")
    print("  R: Restart")
    print("  💡 Game auto-moves every 0.3 seconds")


def play_auto_text_game(grid_size=15):
    """Play Snake game with automatic movement every 0.3 seconds."""
    env = SnakeGameEnv(grid_size=grid_size)
    env.reset()
    
    # Map keys to actions
    key_map = {
        'w': 0, 's': 2, 'a': 3, 'd': 1,
        'W': 0, 'S': 2, 'A': 3, 'D': 1
    }
    
    last_action = 1  # Default RIGHT direction
    game_over = False
    
    
    def input_thread():
        """Thread to handle input without blocking."""
        nonlocal game_over, last_action
        while not game_over:
            try:
                user_input = getch().lower()
                if user_input == 'q':
                    game_over = True
                    break
                elif user_input == 'r':
                    env.reset()
                    last_action = 1
                    clear_screen()
                    print_game_state(env, last_action)
                    print("🔄 Game restarted!")
                elif user_input in key_map:
                    last_action = key_map[user_input]
            except Exception:
                game_over = True
                break
    
    # Start input thread
    input_thread_obj = threading.Thread(target=input_thread, daemon=True)
    input_thread_obj.start()
    
    try:
        while not game_over:
            clear_screen()
            print_game_state(env, last_action)
            
            # Take action
            _, _, terminated, truncated, info = env.step(last_action)
            
            if terminated or truncated:
                clear_screen()
                print_game_state(env, last_action)
                print(f"\n💀 Game Over! Final Score: {info['score']}")
                
                # Ask for restart
                print("\nPlay again? (R:restart, Q:quit): ", end="", flush=True)
                while True:
                    choice = getch().lower()
                    if choice == 'r':
                        env.reset()
                        last_action = 1
                        break
                    elif choice == 'q':
                        game_over = True
                        break
                if game_over:
                    break
                continue
                    
            # Auto-move every 0.3 seconds
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\n🛑 Game ended by user")
    finally:
        game_over = True
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-step Snake game in terminal')
    parser.add_argument('--grid-size', type=int, default=15,
                       help='Grid size for the game (default: 15)')
    
    args = parser.parse_args()
    
    try:
        play_auto_text_game(args.grid_size)
    except KeyboardInterrupt:
        print("\nGame ended by user")
        sys.exit(0)