import os
from pydub import AudioSegment


if __name__ == "__main__":
  folder_path = 'datasets/final_final'

  max_length = 0
  max_len_file = ""
  for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):
      file_path = os.path.join(folder_path, file_name)
      audio = AudioSegment.from_file(file_path)
      duration = len(audio) / 1000  # Convert to seconds
      if duration > max_length:
        max_length = duration
        max_len_file = file_name

  print(f"Longest audio {max_len_file} | {max_length}seconds")
