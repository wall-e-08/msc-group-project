import os
from random import randint
from pytube import YouTube
from moviepy.editor import VideoFileClip


def download_youtube_audio(url):
  output_file = f"datasets/{randint(10000, 99999)}.wav"
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
  yt_url = [
    "https://www.youtube.com/watch?v=bVNwYznbBdM",
  ]
  for _url in yt_url:
    download_youtube_audio(_url)
