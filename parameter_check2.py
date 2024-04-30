import os
import math
import librosa
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = "datasets/final_final"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

MFCC_NUM = [10,15,20,25,30,35,40]
FFT_NUM = [2**10, 2**11, 2**12]
HOP_LENGTH = [2**8, 2**9, 2**10, 2**11]
SEGMENTS_NUM = [8,10,12]

category_labelling_map = {
  "Bird": 0,
  "Engine": 1,
  "Explosion": 2,
  "Human": 3,
  "Rail-transport": 4,
  "Vehicle": 5,
}

def prep_to_get_Xy(mfcc_num, fft_num, hop_len, seg_num):
  mfcc_feature = {
    "category": [],
    "labels": [],
    "mfcc": []
  }

  samples_per_segment = int(SAMPLES_PER_TRACK / seg_num)
  num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_len)

  for i, f in enumerate(os.listdir(DATASET_PATH)):
    file_path = os.path.join(DATASET_PATH, f)
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # number of segments based on duration
    num_segments = int(len(signal) / SAMPLES_PER_TRACK) + 1

    category = f.split("_")[0]
    mfcc_feature["category"].append(category)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    for d in range(num_segments):
      start = samples_per_segment * d
      finish = start + samples_per_segment

      # extract mfcc
      mfcc = librosa.feature.mfcc(y=signal[start:finish],
                                  sr=sample_rate, n_mfcc=mfcc_num, n_fft=fft_num,
                                  hop_length=hop_len)
      mfcc = mfcc.T

      # store only mfcc feature with expected number of vectors
      if len(mfcc) == num_mfcc_vectors_per_segment:
        mfcc_feature["mfcc"].append(mfcc.tolist())
        mfcc_feature["labels"].append(category_labelling_map[category])

  return {
    "X": np.array(mfcc_feature["mfcc"]),
    "y": np.array(mfcc_feature["labels"])
  }

def get_alg_accuracy(X, y):
  samples, frames, ft_per_frame = X.shape
  X_reshaped = X.reshape(samples, -1)

  X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y,
                                                      test_size=0.2, random_state=42)

  svm_clf = SVC(kernel='linear', C=2.0)
  svm_clf.fit(X_train, y_train)
  svm_predictions = svm_clf.predict(X_test)
  svm_accuracy = accuracy_score(y_test, svm_predictions)

  lr_clf = LogisticRegression(max_iter=2000)
  lr_clf.fit(X_train, y_train)
  lr_predictions = lr_clf.predict(X_test)
  lr_accuracy = accuracy_score(y_test, lr_predictions)

  dt_clf = DecisionTreeClassifier()
  dt_clf.fit(X_train, y_train)
  dt_predictions = dt_clf.predict(X_test)
  dt_accuracy = accuracy_score(y_test, dt_predictions)

  return {
    "SVM": svm_accuracy,
    "LR": lr_accuracy,
    "DT": dt_accuracy
  }

if __name__ == '__main__':
  combinations = product(MFCC_NUM, FFT_NUM, HOP_LENGTH, SEGMENTS_NUM)
  for index, combination in enumerate(combinations):
    result = f"{index}. "
    print(f"Starting: {index}")


    Xy = prep_to_get_Xy(mfcc_num=combination[0], fft_num=combination[1],
                        hop_len=combination[2], seg_num=combination[3])
    accuracy = get_alg_accuracy(X=Xy["X"], y=Xy["y"])
    result += f"mfcc: {combination[0]}, fft: {combination[1]}, hop: {combination[2]}, seg: {combination[3]} || Accuracy: {accuracy}"

    with open("result2.txt", "a") as file:
      file.write(result + "\n")
