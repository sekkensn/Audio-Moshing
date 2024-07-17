import os
import random
import librosa
from pydub import AudioSegment
from tqdm import tqdm  # Import tqdm for the loading bar

# Define paths for the original and edited files directories
original_files_dir = "original_files"
edited_files_dir = "edited_files"

# Ensure the edited_files directory exists
if not os.path.exists(edited_files_dir):
    os.makedirs(edited_files_dir)

# Function to process audio files
def process_audio_file(file_path, output_dir, output_file_name, beats_per_slice):
    try:
        y, sr = librosa.load(file_path)
        print(f"Processing {file_path}...")

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        print(f"Detected BPM: {tempo}")

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times_ms = [int(bt * 1000) for bt in beat_times]

        file_name, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower().replace('.', '')
        audio = AudioSegment.from_file(file_path, format=file_extension)

        slices = []
        for i in range(0, len(beat_times_ms) - 1, beats_per_slice):
            start_ms = beat_times_ms[i]
            end_ms = beat_times_ms[min(i + beats_per_slice, len(beat_times_ms) - 1)]
            slices.append(audio[start_ms:end_ms])

        random.shuffle(slices)
        reordered_audio = sum(slices)

        output_file_path = os.path.join(output_dir, output_file_name + ".wav")
        reordered_audio.export(output_file_path, format="wav")
        print(f"Exported reordered file to {output_file_path}")

        return len(slices)  # Return the number of slices processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

# Process each file in the original_files directory
for file_name in os.listdir(original_files_dir):
    file_path = os.path.join(original_files_dir, file_name)
    
    if os.path.isfile(file_path):
        custom_output_name = input(f"Enter the custom name for the edited file (without extension) for {file_name}: ")
        beats_per_slice = int(input(f"Enter number of beats per slice for {file_name}: "))

        # Use tqdm for the loading bar
        with tqdm(desc=f"Processing {file_name}", unit=" slice", colour="green") as pbar:
            try:
                total_slices = process_audio_file(file_path, edited_files_dir, custom_output_name, beats_per_slice)
                if total_slices > 0:
                    pbar.total = total_slices  # Set total slices for tqdm
                    pbar.update(total_slices)  # Update progress bar to completion
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                pbar.update(0)

print("All files processed.")
