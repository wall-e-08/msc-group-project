import os
import subprocess


if __name__ == '__main__':
  files_dir = os.path.abspath("datasets")

  for root, dirs, files in os.walk(files_dir):
    for f in files:
      _path = os.path.join(root, f)
      _base, _ext = os.path.splitext(f)

      output_file_path = os.path.join(files_dir, _base + ".wav")
      if _ext == '.mp4':
        print(f"Converting {_base} to .wav")
        status, output = subprocess.getstatusoutput(f'ffmpeg -i "{_path}" "{output_file_path}"')
        if status:
          print(output)

