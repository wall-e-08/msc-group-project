import os
import random
import re
import json
from moviepy.editor import VideoFileClip
from pytube import YouTube
from pydub import AudioSegment


json_file = "google_mediapipe_ontology.json"

def download_youtube_audio_segment(url, file_name, start_time, end_time, index):
  output_file = f"datasets/others/{file_name}_{index+1}.wav"
  tmp_audio = "datasets/others/temp.wav"
  try:
    # download youtube video
    yt = YouTube(url)
    (yt.streams
      .filter(progressive=True, file_extension='mp4')
      .first()
      .download(filename="temp.mp4"))

    # Convert to WAV
    video_clip = VideoFileClip('temp.mp4')

    video_clip.audio.write_audiofile(tmp_audio)

    audio = AudioSegment.from_wav(tmp_audio)
    # output_file_path = output_file
    # output_file = output_file.replace(".wav", f"{random.randint(1000,9999)}.wav")
    output_file_path = os.path.join(os.getcwd(), output_file)
    segment = audio[start_time:end_time]
    segment.export(output_file_path, format="wav")
    print("Audio saved successfully in:", output_file_path)
  except Exception as e:
    print("Error:", str(e))
  finally:
    # Clean up temporary files
    if os.path.exists('temp.mp4'):
      os.remove('temp.mp4')
    if os.path.exists(tmp_audio):
      os.remove(tmp_audio)

def extract_links_from_examples(examples):
  links = []
  for example in examples:
    match = re.search(r"(youtu\.be\/[\w\-]+)\?start=(\d+)&end=(\d+)", example)
    if match:
      links.append({
        "link": match.group(1),
        "start": int(match.group(2)),
        "end": int(match.group(3))
      })
  return links

def extract_yt_links_from_json(category_name):
  with open(json_file, "r") as f:
    data = json.load(f)

  links = []

  # Find the category by name
  category = next((item for item in data if item["name"] == category_name), None)
  if category:
    # Extract positive examples from the category
    links.extend(extract_links_from_examples(category["positive_examples"]))

    # Recursively extract positive examples from children
    for child_id in category["child_ids"]:
      child_category = next((item for item in data if item["id"] == child_id), None)
      if child_category:
        links.extend(extract_links_from_examples(child_category["positive_examples"]))

  return links

if __name__ == "__main__":
  for category in ["Vehicle", "Rail transport", "Engine",
                   "Male speech, man speaking", "Female speech, woman speaking",
                   "Bird", "Screaming", "Train horn",
                   "Explosion", "Child speech, kid speaking"
                   ,"Conversation", "Narration, monologue", "Babbling", "Speech synthesizer"]:
    result = extract_yt_links_from_json(category)
    for index, item in enumerate(result):
      download_youtube_audio_segment(
        url=item["link"],
        file_name=f'{category.replace(", ", "_").replace(" ", "_").replace(",", "_")}',
        start_time=item["start"] * 1000,
        end_time=item["end"] * 1000,
        index=index,
      )