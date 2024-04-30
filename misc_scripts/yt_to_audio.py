import os
from random import randint
from pytube import YouTube
from moviepy.editor import VideoFileClip


def download_youtube_audio(url, category):
  output_file = f"datasets/yt/{category}_{randint(10000, 99999)}.wav"
  try:
    # download youtube video
    yt = YouTube(url)
    (yt.streams
      .filter(progressive=True, file_extension='mp4')
      .first()
      .download(filename="temp.mp4"))

    # Convert to WAV
    video_clip = VideoFileClip('temp.mp4')
    video_clip.audio.write_audiofile(output_file)

    print("Audio saved successfully as:", output_file)
  except Exception as e:
    print("Error:", str(e))
  finally:
    # Clean up temporary files
    if os.path.exists('temp.mp4'):
      os.remove('temp.mp4')


if __name__ == "__main__":
  yt_url = {
    # "ambulance: "https://www.youtube.com/watch?v=bVNwYznbBdM",
    "train_horn": "https://www.youtube.com/watch?v=k5t60QwM-3o",
  }
  for catg, _url in yt_url.items():
    download_youtube_audio(_url, catg)
