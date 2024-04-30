import os
import math
import librosa
import numpy as np


mapping = {
  "Ambulance": 0,
  "Bird": 1,
  "Engine": 2,
  "Explosion": 3,
  "Fire-and-trucks": 4,
  "Human": 5,
  "Police": 6,
  "Rail-transport": 7,
  "Train-horn": 8,
  "Vehicle": 9,
}

def get_mfcc_spectogram(_path, _type, sr=22050, fft=4096, hl=4096):
  features = {
    "category": [],
    "labels": [],
    "data": []
  }

  samples_per_trk = sr * 30

  samples_per_segment = int(samples_per_trk / 10)
  num_per_segment = math.ceil(samples_per_segment / hl)

  for i, f in enumerate(os.listdir(_path)):
    file_path = os.path.join(_path, f)
    signal, sample_rate = librosa.load(file_path, sr=sr)

    # number of segments based on duration
    num_segments = int(len(signal) / samples_per_trk) + 1

    category = f.split("_")[0]
    features["category"].append(category)

    checker = 1

    for d in range(num_segments):
      start = samples_per_segment * d
      finish = start + samples_per_segment

      _data = librosa.feature.mfcc(y=signal[start:finish],
                                   sr=sample_rate, n_fft=fft,
                                   hop_length=hl)
      if _type == "spectogram_db":
        _data = librosa.power_to_db(_data)

      _data = _data.T

      # store feature only with expected number of vectors
      if len(_data) == num_per_segment:
        features["data"].append(_data.tolist())
        features["labels"].append(mapping[category])
      else:
        checker += 1
      if checker != 0:
        print(f"Audio segments overload: {f}, {checker=}, {i=}")
  return features


mfcc_feature = get_mfcc_spectogram("datasets/final_final", "mfcc")

unique_elements, counts = np.unique(mfcc_feature["labels"], return_counts=True)

for element, count in zip(unique_elements, counts):
  print(f"{element} : {count}")