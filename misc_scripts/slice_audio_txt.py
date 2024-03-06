import os

from pydub import AudioSegment


audio_categories = ["ambulance", "fire-and-trucks", "police"]
data_dir_prefix = "datasets/vens"

for catg in audio_categories:
  audio = AudioSegment.from_wav(f"{data_dir_prefix}/{catg}.wav")

  with open(f"{data_dir_prefix}/{catg}.txt", 'r') as catg_file:
    for line_number, line in enumerate(catg_file, start=1):
      start_time_str, end_time_str = line.strip().split('-')

      start_minutes, start_seconds = map(int, start_time_str.split(':'))
      end_minutes, end_seconds = map(int, end_time_str.split(':'))

      start_time = start_minutes * 60 * 1000 + start_seconds * 1000
      end_time = end_minutes * 60 * 1000 + end_seconds * 1000

      segment = audio[start_time:end_time]

      output_file = f"{catg}_{line_number}.wav"
      output_file_path = os.path.join(os.getcwd(), "datasets/final_audio/", output_file)
      segment.export(output_file_path, format="wav")

      print(f"Saved: {output_file}")

print("All segments created successfully.")


