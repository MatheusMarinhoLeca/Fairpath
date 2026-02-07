import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + "\n")

def get_user_choice(options, prompt="Select an option"):
    for key, value in options.items():
        print(f"[{key}] {value}")
    
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in options:
            return choice
        print("Invalid selection. Please try again.")

def get_user_confirmation(prompt="Are you sure?"):
    while True:
        choice = input(f"{prompt} (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'.")

def get_user_input(prompt, validator=None, allow_empty=False):
    while True:
        user_input = input(f"{prompt}: ").strip()
        if not user_input:
            if allow_empty:
                return user_input
            print("Input cannot be empty.")
            continue
        
        if validator:
            if validator(user_input):
                return user_input
            else:
                print("Invalid input. Please check your entry.")
        else:
            return user_input
