from openai import OpenAI
import sounddevice as sd
import numpy as np
import tempfile
import pygame
import threading
import random
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import yaml
import cv2
import base64
import os
import requests
import concurrent.futures
import sys
from queue import Queue
import ollama
from groq import Groq
from io import BytesIO
from pydub.playback import play
import queue
from scipy.io import wavfile
import time
from ohbot import ohbot

eleven_client = ElevenLabs(
    api_key=os.environ.get("ELEVEN_API_KEY")
)

# Get your OpenAI API Key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the filename is provided as a command-line argument
if len(sys.argv) < 2:
    print("Error: No YAML file name provided.")
    sys.exit(1)

filename = sys.argv[1]

try:
    with open(filename, 'r') as file:
        settings = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: The file {filename} was not found.")
    sys.exit(1)

sampling_rate = settings['sampling_rate']
num_channels = settings['num_channels']
dtype = np.dtype(settings['dtype'])  # Convert string to numpy dtype
silence_threshold = settings['silence_threshold']
ambient_noise_level_threshold_multiplier = settings['ambient_noise_level_threshold_multiplier']
max_file_size_bytes = settings['max_file_size_bytes']
enable_lonely_sounds = settings['enable_lonely_sounds']
enable_squeak = settings['enable_squeak']
system_prompt = settings['system_prompt']
voice_id = settings['voice_id']
greetings = settings['greetings']
lonely_sounds = settings['lonely_sounds']
enable_vision = settings['enable_vision']
silence_count_threshold = settings['silence_count_threshold']
model = settings['model']

# Initialize the client based on the model
if model.startswith("gpt"):
    client = OpenAI()
elif model.startswith("groq"):
    groqClient = Groq()
else:
    ollama_client = ollama.Client()

# Initialize messages
messages = [
    {"role": "system", "content": system_prompt}
]

# Function to get a random greeting
def get_random_greeting():
    return random.choice(greetings)

def play_lonely_sound():
    global silence_threshold  # Access the global silence_threshold
    if not talking:
        original_silence_threshold = silence_threshold
        silence_threshold = 1000  # Increase threshold temporarily
        lonely_file = random.choice(lonely_sounds)
        print(f"Playing lonely sound: {lonely_file}")
        pygame.mixer.music.load(lonely_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass  # Wait for the sound to finish playing
        silence_threshold = original_silence_threshold  # Reset to original threshold
    else:
        print("Not playing lonely sound because the mouse is talking.")

def calculate_ambient_noise_level(duration=5):
    print("Calculating ambient noise level, please wait...")
    ambient_noise_data = []
    for _ in range(int(duration / duration)):  # duration / duration = number of chunks
        audio_chunk = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=num_channels, dtype=dtype)
        sd.wait()
        ambient_noise_data.extend(audio_chunk)
    ambient_noise_level = np.abs(np.array(ambient_noise_data)).mean()
    print(f"Ambient noise level: {ambient_noise_level}")

    return ambient_noise_level

def get_pet_reply(user_input, base64_image=None):
    global messages
    
    # Prepare the user message
    if enable_vision and base64_image:
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    else:
        user_message = {"role": "user", "content": user_input}
    
    messages.append(user_message)
    
    # If model starts with "gpt" then use the chat endpoint from OpenAI
    if model.startswith("gpt"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content
    
    elif model.startswith("groq"):
        # Remove "grok-" from the model name
        model_name = model[5:]
        response = groqClient.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content

    else: # Use ollama local deployment
        print(f"Using Ollama model: {model}")
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']

def get_emotion_movements(text):
    """Analyze text and return appropriate movement sequence"""
    # Convert to lowercase for easier matching
    text = text.lower()
    
    # Keywords for different emotions/responses
    agreement_words = ['yes', 'agree', 'correct', 'right', 'absolutely', 'indeed']
    disagreement_words = ['no', 'disagree', 'incorrect', 'wrong', "don't think so"]
    thinking_words = ['hmm', 'well', 'let me think', 'perhaps', 'maybe']
    excited_words = ['wow', 'amazing', 'awesome', 'excellent', 'fantastic']
    
    def perform_movement_sequence():
        if any(word in text for word in agreement_words):
            print("Performing agreement nod")
            for _ in range(2):  # Nod twice
                ohbot.move(ohbot.HEADNOD, 8, 5)  # Move down
                ohbot.wait(0.2)
                ohbot.move(ohbot.HEADNOD, 3, 5)  # Move up
                ohbot.wait(0.2)
        
        elif any(word in text for word in disagreement_words):
            print("Performing disagreement shake")
            for _ in range(2):  # Shake twice
                ohbot.move(ohbot.HEADTURN, 7, 5)  # Turn right
                ohbot.wait(0.2)
                ohbot.move(ohbot.HEADTURN, 3, 5)  # Turn left
                ohbot.wait(0.2)
        
        elif any(word in text for word in thinking_words):
            print("Performing thinking motion")
            ohbot.move(ohbot.HEADTURN, 6, 3)  # Slight turn
            ohbot.move(ohbot.EYETILT, 7, 3)   # Look up
            ohbot.wait(0.5)
            ohbot.move(ohbot.LIDBLINK, 10, 5)  # Blink slowly
            ohbot.wait(0.3)
        
        elif any(word in text for word in excited_words):
            print("Performing excited motion")
            ohbot.move(ohbot.HEADNOD, 3, 10)  # Quick head up
            ohbot.move(ohbot.EYETILT, 3, 10)  # Eyes wide
            ohbot.wait(0.2)
            ohbot.move(ohbot.LIDBLINK, 0, 10)  # Eyes wide open
            ohbot.wait(0.3)
        
        # Always return to center position
        ohbot.move(ohbot.HEADTURN, 5, 3)
        ohbot.move(ohbot.HEADNOD, 5, 3)
        ohbot.move(ohbot.EYETURN, 5, 3)
        ohbot.move(ohbot.EYETILT, 5, 3)
        ohbot.wait(0.1)

    # Run the movement sequence in a separate thread to not block
    movement_thread = threading.Thread(target=perform_movement_sequence)
    movement_thread.start()
    return movement_thread

def say(text):
    global messages

    if enable_squeak:
        pygame.mixer.music.stop()

    # Start emotion-based movements before speaking
    movement_thread = get_emotion_movements(text)

    # Generate audio stream for the assistant's reply
    audio_stream = eleven_client.generate(
        voice=voice_id,
        text=text,
        model="eleven_turbo_v2",
        stream=True
    )

    # Create a BytesIO object to hold audio data
    audio_buffer = BytesIO()

    # Stream the generated audio
    for chunk in audio_stream:
        if chunk:
            audio_buffer.write(chunk)

    # Reset buffer position to the beginning
    audio_buffer.seek(0)

    # Use timestamp for unique filename in temp directory
    timestamp = int(time.time())
    output_filename = os.path.join('temp', f'pet_response_{timestamp}.mp3')
    
    try:
        # Save the audio buffer to an MP3 file
        with open(output_filename, "wb") as f:
            f.write(audio_buffer.getvalue())

        print(f"Saved audio response to {output_filename}")

        # Display the talking pet while playing the audio
        display_talking_pet(output_filename)

        messages.append({"role": "assistant", "content": text})

        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]
    finally:
        # Clean up the audio file
        try:
            if os.path.exists(output_filename):
                os.remove(output_filename)
        except Exception as e:
            print(f"Error cleaning up audio file: {e}")

    # Wait for movement sequence to complete before continuing
    movement_thread.join()

def get_user_input_from_audio(audio_data):
    global talking

    # Create temp file in temp directory
    temp_filename = os.path.join('temp', 'temp_audio.mp3')
    try:
        # Create the audio segment
        audio_segment = AudioSegment(
            data=np.array(audio_data).tobytes(),
            sample_width=dtype.itemsize,
            frame_rate=sampling_rate,
            channels=num_channels
        )
        # Export directly to the temp file
        audio_segment.export(temp_filename, format="mp3")

        talking = True

        print(f"File name: {temp_filename}")

        if enable_squeak:
            pygame.mixer.music.play()

        # Transcribe audio using OpenAI API
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcript.text
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")

def initialize():
    global silence_threshold
    
    print("Initializing Ohbot...")
    ohbot.init()
    
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
        print("Created temp directory")
    
    # Clean up any old temp files
    cleanup_temp_files()
    
    # Center all movements initially
    ohbot.move(ohbot.HEADTURN, 5)
    ohbot.move(ohbot.HEADNOD, 5)
    ohbot.move(ohbot.EYETURN, 5)
    ohbot.move(ohbot.EYETILT, 5)
    ohbot.move(ohbot.LIDBLINK, 0)
    ohbot.move(ohbot.TOPLIP, 5)
    ohbot.move(ohbot.BOTTOMLIP, 5)
    ohbot.wait(0.5)
    
    # Calculate ambient noise level and set it as the silence threshold
    ambient_noise_level = calculate_ambient_noise_level()
    silence_threshold = ambient_noise_level * ambient_noise_level_threshold_multiplier
    
    # Set silence_threshold to a minimum value
    silence_threshold = max(silence_threshold, 10.0)
    
    print(f"Silence threshold: {silence_threshold}")
    
    if enable_lonely_sounds:
        # Initialize the periodic lonely sound timer
        timer = threading.Timer(60, play_lonely_sound)
        timer.start()

def cleanup_temp_files():
    """Clean up old temporary files"""
    temp_dir = 'temp'
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            if file.startswith('pet_response_'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    print(f"Error removing temp file {file}: {e}")

def analyze_audio_chunk(audio_chunk):
    samples = np.array(audio_chunk.get_array_of_samples())
    if len(samples) == 0:
        return False
    rms = np.sqrt(np.mean(samples.astype(float)**2))
    mouth_open = rms > 50  # Lowered threshold, adjust as needed
    print(f"RMS: {rms:.2f}, Mouth open: {mouth_open}")  # Debug print
    return mouth_open

def analyze_audio_file(audio_file):
    # Convert mp3 to wav
    audio = AudioSegment.from_mp3(audio_file)
    audio.export("temp.wav", format="wav")
    
    # Read the wav file
    sample_rate, audio_data = wavfile.read("temp.wav")
    
    # If stereo, convert to mono
    if len(audio_data.shape) == 2:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Calculate energy with smaller frame size for more frequent updates
    frame_length = int(sample_rate * 0.02)  # 20ms frames
    energy = []
    for i in range(0, len(audio_data), frame_length):
        frame = audio_data[i:i+frame_length]
        energy.append(np.sum(frame**2))
    
    # Normalize energy
    energy = np.array(energy)
    energy = energy / np.max(energy)
    
    return energy, len(audio_data) / sample_rate

def play_and_analyze_audio(audio_file, mouth_queue):
    print("Starting audio analysis")
    energy, duration = analyze_audio_file(audio_file)
    
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Send mouth states to the queue
    start_time = time.time()
    for e in energy:
        mouth_open = e > 0.03  # More sensitive threshold
        mouth_queue.put(mouth_open)
        print(f"Energy: {e:.2f}, Mouth open: {mouth_open}")
        
        # Wait for precise timing
        elapsed = time.time() - start_time
        target_time = elapsed + 0.02
        while time.time() < target_time:
            time.sleep(0.001)
        
        # Check if audio finished playing
        if not pygame.mixer.music.get_busy():
            break
    
    # Signal end of audio
    mouth_queue.put(None)
    print("Audio analysis complete")

def resize_image(image, window_size):
    """Resize image to fit the window while maintaining aspect ratio."""
    img_w, img_h = image.get_size()
    win_w, win_h = window_size
    aspect_ratio = img_w / img_h
    if win_w / win_h > aspect_ratio:
        new_h = win_h
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = win_w
        new_h = int(new_w / aspect_ratio)
    return pygame.transform.smoothscale(image, (new_w, new_h))

def random_head_movement():
    while True:
        try:
            # Random movement type (0: normal, 1: look around, 2: focused look)
            movement_type = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
            
            if movement_type == 0:  # Normal subtle movements
                head_turn = random.uniform(4, 6)    # Subtle turn around center position
                head_nod = random.uniform(4, 6)     # Subtle nod around center position
                eye_turn = random.uniform(4, 6)     # Subtle eye turn
                eye_tilt = random.uniform(4, 6)     # Subtle eye tilt
            
            elif movement_type == 1:  # Look around
                # More extreme movements for "looking around" behavior
                head_turn = random.uniform(3, 7)
                eye_turn = head_turn + random.uniform(-1, 1)  # Eyes follow head movement
                head_nod = random.uniform(3, 7)
                eye_tilt = head_nod + random.uniform(-1, 1)
            
            else:  # Focused look at something
                # Eyes move first, then head follows
                eye_turn = random.uniform(2, 8)
                eye_tilt = random.uniform(2, 8)
                ohbot.move(ohbot.EYETURN, eye_turn, 10)
                ohbot.move(ohbot.EYETILT, eye_tilt, 10)
                ohbot.wait(0.2)
                head_turn = eye_turn + random.uniform(-0.5, 0.5)
                head_nod = eye_tilt + random.uniform(-0.5, 0.5)
            
            # Randomly decide if we should blink
            should_blink = random.random() < 0.3  # 30% chance to blink
            
            # Execute movements
            ohbot.move(ohbot.HEADTURN, head_turn, 2)
            ohbot.move(ohbot.HEADNOD, head_nod, 2)
            ohbot.move(ohbot.EYETURN, eye_turn, 3)  # Slightly faster eye movements
            ohbot.move(ohbot.EYETILT, eye_tilt, 3)
            
            # If blinking, do a quick blink motion
            if should_blink:
                print("Blinking...")
                ohbot.move(ohbot.LIDBLINK, 10, 10)  # Close eyes quickly
                ohbot.wait(0.1)
                ohbot.move(ohbot.LIDBLINK, 0, 10)   # Open eyes quickly
            
            # Random wait between movements
            if movement_type == 0:
                time.sleep(random.uniform(1, 3))
            elif movement_type == 1:
                time.sleep(random.uniform(0.5, 1.5))  # Shorter pauses when looking around
            else:
                time.sleep(random.uniform(2, 4))  # Longer pauses when focused
            
        except Exception as e:
            print(f"Head movement error: {e}")
            break

def display_talking_pet(audio_file):
    print("Starting display_talking_pet function")

    # Start the random head movement in a separate thread
    head_movement_thread = threading.Thread(target=random_head_movement, daemon=True)
    head_movement_thread.start()

    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Advanced mouth movement pattern with variability
    while pygame.mixer.music.get_busy():
        # Randomize mouth opening amount (between 7-9 for top lip, 7-9 for bottom lip)
        top_open = random.uniform(7, 9)
        bottom_open = random.uniform(7, 9)
        
        print("Opening mouth")
        ohbot.move(ohbot.TOPLIP, top_open)
        ohbot.move(ohbot.BOTTOMLIP, bottom_open)
        
        # Random wait time for open position (0.1-0.3 seconds)
        ohbot.wait(random.uniform(0.1, 0.3))
        
        print("Closing mouth")
        # Vary the "closed" position slightly (4.5-5.5)
        closed_pos = random.uniform(4.5, 5.5)
        ohbot.move(ohbot.TOPLIP, closed_pos)
        ohbot.move(ohbot.BOTTOMLIP, closed_pos)
        
        # Random wait time for closed position (0.1-0.25 seconds)
        ohbot.wait(random.uniform(0.1, 0.25))

    # Ensure mouth is closed at the end
    print("Closing mouth - end of audio")
    ohbot.move(ohbot.TOPLIP, 5)
    ohbot.move(ohbot.BOTTOMLIP, 5)
    ohbot.wait(0.1)

# Global list to accumulate audio data
audio_queue = Queue()
silence_count = 0
stop_recording = False
first_sound_detected = False
total_frames = 0

def audio_callback(indata, frames, time, status):
    global audio_queue, silence_count, stop_recording, first_sound_detected, total_frames
    
    # Compute the mean amplitude of the audio chunk
    chunk_mean = np.abs(indata).mean()
    print(f"Chunk mean: {chunk_mean}")  # Debugging line
    
    if chunk_mean > silence_threshold:
        
        print("Sound detected, adding to audio queue.")
        audio_queue.put(indata.copy())
        total_frames += frames

        silence_count = 0
        first_sound_detected = True

    elif first_sound_detected:

        silence_count += 1
    
    # Print silence count
    print(f"Silence count: {silence_count}")

    # Stop recording after a certain amount of silence, but make sure it's at least 0.25 second long.
    current_duration = total_frames / sampling_rate

    if current_duration >= 0.25 and silence_count >= silence_count_threshold and first_sound_detected:
        stop_recording = True
        print("Silence detected, stopping recording.")

def process_audio_from_queue():

    print("Processing audio data...")

    audio_data = np.empty((0, num_channels), dtype=dtype)

    while not audio_queue.empty():
        # Get data from the queue
        indata = audio_queue.get()
 
        # Append the chunk to the audio data
        audio_data = np.concatenate((audio_data, indata), axis=0)

    return audio_data

def listen_audio_until_silence():
    global audio_queue, silence_count, stop_recording, first_sound_detected, total_frames

    # Reset audio data and silence count
    print("Listening for audio...")
    audio_queue = Queue()
    silence_count = 0
    stop_recording = False
    first_sound_detected = False
    total_frames = 0

    # Print available devices
    print("Available devices:")
    print(sd.query_devices())

    try:
        # Start recording audio
        with sd.InputStream(callback=audio_callback, samplerate=sampling_rate, channels=num_channels, dtype=dtype):
            while not stop_recording:
                sd.sleep(250)  # Sleep for a short time to prevent busy waiting

        print("Recording stopped.")

        # Process audio data from the queue
        audio_data = process_audio_from_queue()

        return audio_data

    except sd.PortAudioError as e:
        print(f"PortAudio error: {e}")
        print(f"Current audio settings:")
        print(f"Sampling rate: {sampling_rate}")
        print(f"Channels: {num_channels}")
        print(f"Dtype: {dtype}")
        return None

def capture_webcam_as_base64():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    
    # Save the frame as an image file
    cv2.imwrite('view.jpeg', frame)

    retval, buffer = cv2.imencode('.jpg', frame)
    if not retval:
        print("Failed to encode image.")
        return None

    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def test_ohbot():
    print("Testing Ohbot movements...")
    ohbot.init()
    
    try:
        for _ in range(3):  # Test 3 times
            print("Opening mouth wide")
            ohbot.move(ohbot.TOPLIP, 9)
            ohbot.move(ohbot.BOTTOMLIP, 1)
            ohbot.wait(1)
            
            print("Closing mouth")
            ohbot.move(ohbot.TOPLIP, 5)
            ohbot.move(ohbot.BOTTOMLIP, 5)
            ohbot.wait(1)
    finally:
        ohbot.close()

def main():
    global talking, messages, stop_recording

    initialize()
    
    try:
        say(get_random_greeting())

        print("Listening...")

        while True:
            talking = False

            audio_data = listen_audio_until_silence()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_user_input = executor.submit(get_user_input_from_audio, audio_data)
                
                if enable_vision:
                    future_pet_view = executor.submit(capture_webcam_as_base64)

                user_input = future_user_input.result()
                base64_image = future_pet_view.result() if enable_vision else None

            print(f"User: {user_input}")

            pet_reply = get_pet_reply(user_input, base64_image)
            print(f"Pet: {pet_reply}")

            if pet_reply.lower() == "ignore":
                print("Ignoring conversation...")
                messages = messages[:-1]
            else:
                say(pet_reply)

    finally:
        ohbot.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        ohbot.close()