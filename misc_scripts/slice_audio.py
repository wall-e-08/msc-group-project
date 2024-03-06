import os

import pandas as pd
from pydub import AudioSegment

sheet_name = ["Ambulances 1", "Fire Engines and Trucks respond", "British Transport Police respon"]
audio_file_prefix = {
  "Ambulances 1": "ambulance",
  "Fire Engines and Trucks respond": "fire",
  "British Transport Police respon": "police"
}
dfs = pd.read_excel("datasets/vens/data.xlsx", sheet_name=sheet_name)

audio = AudioSegment.from_wav("datasets/vens/British Transport Police responding with siren and lights (Compilation).wav")

for sheet in sheet_name:
  for index, row in dfs[sheet].iterrows():
    segment_id = row['ID']
    start_time_str = row['Start']
    end_time_str = row['End']

    # Time string to milliseconds
    start_minutes, start_seconds = map(int, start_time_str.split(':'))
    end_minutes, end_seconds = map(int, end_time_str.split(':'))

    start_time = start_minutes * 60 * 1000 + start_seconds * 1000
    end_time = end_minutes * 60 * 1000 + end_seconds * 1000

    segment = audio[start_time:end_time]

    output_file = f"{audio_file_prefix[sheet]}_{segment_id}.wav"
    output_file_path = os.path.join(os.getcwd(), "datasets/final_audio/", output_file)
    segment.export(output_file_path, format="wav")

    print(f"Segment {segment_id} saved as {output_file}")

print("All segments created successfully.")
