import os
import math
import librosa
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


DATASET_PATH = "datasets/final_final"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

MFCC_NUM = [10,30]
FFT_NUM = [2**8, 2**10, 2**12]
HOP_LENGTH = [2**8, 2**10, 2**12]
SEGMENTS_NUM = [5,10]
BATCH_SIZE = [32,]
EPOCHS = [200,]

category_labelling_map = {
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
    print("-"*5, category)
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

def get_cnn_accuracy(X, y, batch_size, epochs):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # add an axis to input sets
  X_train = X_train[..., np.newaxis]
  X_test = X_test[..., np.newaxis]

  cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),

    Conv2D(32, (2, 2), activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    BatchNormalization(),

    # flatten output
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),

    # output
    Dense(len(category_labelling_map), activation='softmax')
  ])

  cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    # loss='categorical_crossentropy',
                    metrics=['accuracy'])

  cnn_history = cnn_model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=batch_size, epochs=epochs)

  cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=2)
  return cnn_accuracy

if __name__ == '__main__':
  combinations = product(MFCC_NUM, FFT_NUM, HOP_LENGTH, SEGMENTS_NUM, BATCH_SIZE, EPOCHS)
  for index, combination in enumerate(combinations):
    result = f"{index}. "
    print(f"Starting: {index}")

    Xy = prep_to_get_Xy(mfcc_num=combination[0], fft_num=combination[1],
                        hop_len=combination[2], seg_num=combination[3])
    accuracy = get_cnn_accuracy(X=Xy["X"], y=Xy["y"],
                                batch_size=combination[4], epochs=combination[5])
    result += f"mfcc: {combination[0]}, fft: {combination[1]}, hop: {combination[2]}, seg: {combination[3]}, batch: {combination[4]}, epochs: {combination[5]} || Accuracy: {accuracy}"
    with open("result.txt", "a") as file:
      file.write(result + "\n")
    # try:
    #   Xy = prep_to_get_Xy(mfcc_num=combination[0], fft_num=combination[1],
    #                       hop_len=combination[2], seg_num=combination[3])
    #   accuracy = get_cnn_accuracy(X=Xy["X"], y=Xy["y"],
    #                               batch_size=combination[4], epochs=combination[5])
    #   result += f"mfcc: {combination[0]}, fft: {combination[1]}, hop: {combination[2]}, seg: {combination[3]}, batch: {combination[4]}, epochs: {combination[5]} || Accuracy: {accuracy}"
    # except:
    #   result += f"Passing: mfcc: {combination[0]}, fft: {combination[1]}, hop: {combination[2]}, seg: {combination[3]}, batch: {combination[4]}, epochs: {combination[5]}"
    # finally:
    #   with open("result.txt", "a") as file:
    #     file.write(result + "\n")
