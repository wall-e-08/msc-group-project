import os
import wave
import contextlib

def trim_wav(input_file, output_file, start_time, end_time):
    with contextlib.closing(wave.open(input_file, 'r')) as input_wav:
        params = input_wav.getparams()
        sample_width = params.sampwidth
        frame_rate = params.framerate
        num_channels = params.nchannels

        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        input_wav.setpos(start_frame)
        frames_to_read = end_frame - start_frame
        data = input_wav.readframes(frames_to_read)

    with wave.open(output_file, 'w') as output_wav:
        output_wav.setparams((num_channels, sample_width, frame_rate, frames_to_read, params.comptype, params.compname))
        output_wav.writeframes(data)

if __name__ == '__main__':
  audio_folder = "datasets/final_final_22"

  for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
      input_file = os.path.join(audio_folder, audio_file)
      output_file = os.path.join(audio_folder, 'trimmed_' + audio_file)  # Output file will have 'trimmed_' prefix

      trim_wav(input_file, output_file, 0.0, 30)
  pass