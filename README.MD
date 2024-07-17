# Music Moshing

Music Moshing is a Python script that processes audio files by slicing them into segments based on beats and then randomly reorders these segments to create a new audio file. The script supports multiple audio formats such as MP3 and WAV.

## Features

- Detects the tempo (BPM) of the audio file.
- Slices the audio file into segments based on the detected beats.
- Allows customization of the number of beats per slice.
- Randomly reorders the sliced segments.
- Exports the reordered audio file in WAV format.

## Requirements

- Python 3.6+
- `librosa`
- `pydub`
- `tqdm`

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/YOUR_USERNAME/Music-Moshing.git
    cd Music-Moshing
    ```

2. **Install the required Python packages**:

    ```sh
    pip install librosa pydub tqdm
    ```

## Usage

1. **Prepare your audio files**:
    - Place your original audio files in the `original_files` directory.

2. **Run the script**:

    ```sh
    python your_script_name.py
    ```

3. **Follow the prompts**:
    - Enter a custom name for the edited file (without extension).
    - Enter the number of beats per slice.

4. **Find the processed files**:
    - The edited files will be saved in the `edited_files` directory with the custom names you provided.

## Example

1. **Original Files**:
    - Place your MP3, WAV, or other audio files in the `original_files` directory.

2. **Running the Script**:

    ```sh
    python your_script_name.py
    ```

    Follow the prompts:
    - Enter the custom name for the edited file (e.g., `remix1`).
    - Enter the number of beats per slice (e.g., `4`).

3. **Output**:
    - The reordered audio file will be saved as `remix1.wav` in the `edited_files` directory.

## Code Overview

```python
import os
import random
import librosa
from pydub import AudioSegment

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

        try:
            process_audio_file(file_path, edited_files_dir, custom_output_name, beats_per_slice)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("All files processed.")


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
