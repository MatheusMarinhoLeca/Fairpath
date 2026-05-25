import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + "\n")

def get_user_choice(options, title=None, prompt="Select an option"):
    if title:
        print(f"\n{title}:")
    
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

def get_user_input(prompt, validator=None, allow_empty=False, default=None):
    full_prompt = f"{prompt}"
    if default is not None:
        full_prompt += f" [Default: {default}]"
    
    while True:
        user_input = input(f"{full_prompt}: ").strip()
        
        if not user_input:
            if default is not None:
                return default
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
