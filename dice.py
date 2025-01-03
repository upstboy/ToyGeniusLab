import time
import random
import pyttsx3
import winsound
import tkinter as tk

def update_timer(timer_label, remaining_time):
    timer_label.config(text=f"Next in: {remaining_time} seconds")
    timer_label.update()
    
    # Play beep for last 3 seconds
    if remaining_time <= 3:
        winsound.Beep(1000, 200)  # 1000Hz for 200ms

def display_vowel(root, vowel_label, rule_label, timer_label, vowel, rule):
    # Update the text of the vowel and rule labels
    vowel_label.config(text=vowel)
    rule_label.config(text=rule)
    timer_label.config(text="")  # Clear timer text initially
    root.update()

def create_window():
    # Create a window
    root = tk.Tk()
    root.title("Vowel Display")

    # Set window size to full screen
    root.attributes('-fullscreen', True)

    # Set background color
    root.configure(bg='black')

    # Create a label to display the rule
    rule_label = tk.Label(root, text="", font=("Helvetica", 50), fg="white", bg="black")
    rule_label.pack()

    # Create a label to display the vowel
    vowel_label = tk.Label(root, text="", font=("Helvetica", 300), fg="white", bg="black")
    vowel_label.pack(expand=True)

    # Create a label to display the timer
    timer_label = tk.Label(root, text="", font=("Helvetica", 50), fg="yellow", bg="black")
    timer_label.pack()

    return root, vowel_label, rule_label, timer_label

def speak_vowel():
    # Initialize the speech engine
    engine = pyttsx3.init()

    # List of vowels
    vowels = ['A', 'E', 'I', 'O', 'U']

    # List of rules with natural descriptions
    rules = [
        "Generate words where the first letter is",
        "Generate words where the last letter is",
        "Generate words where the vowel appears anywhere",
        "Generate words where the vowel appears twice",
        "Generate words that do not contain",
        "Wild! Make up a creative rule"
    ]

    # Create the window and labels
    root, vowel_label, rule_label, timer_label = create_window()

    while True:
        # Pick a random vowel
        random_vowel = random.choice(vowels)

        # Pick a random rule
        random_rule = random.choice(rules)

        # Beep sound before announcement
        winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

        # Print the vowel and rule to the console
        print(f"Selected rule: {random_rule}, {random_vowel}")

        # Use TTS to speak the vowel and the rule
        engine.say(f"{random_rule}, {random_vowel}.")
        engine.runAndWait()

        # Update the vowel and rule on screen
        display_vowel(root, vowel_label, rule_label, timer_label, random_vowel, random_rule)

        # Wait for a random time between 5 and 30 seconds
        wait_time = random.randint(5, 30)
        for remaining_time in range(wait_time, 0, -1):
            update_timer(timer_label, remaining_time)
            time.sleep(1)

if __name__ == "__main__":
    speak_vowel()
