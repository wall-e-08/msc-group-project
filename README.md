# Sound Source Classification Analysis for Autonomous Vehicle


### Installation
#### Install required libraries (Make sure you have `python` and then `pip` installed)
- `pip install -r requirements.txt`
#### Run jupyter and open browser
- `jupyter notebook`

### Hierarchy
- `google_mediapipe_ontology.json` -> Downloaded from [here](https://github.com/audioset/ontology/blob/master/ontology.json) for youtube video links
- misc_scripts
  - `audio_from_yt_mediapipe.py` -> Using `google_mediapipe_ontology.json` to download youtube videos
  - `yt_to_audio.py` -> Convert youtube videos to audio
  - `video_to_WAV.py` -> Download and convert youtube videos to wav format as required for `librosa` library
  - `slice_audio.py` -> Slice audio using start time and end time
  - `slice_audio_txt.py` -> Slice audio file using text file(<category_name>.txt) containing start time & end time
  - `get_max_audio_length.py` -> calculate maximum audio length from dataset
  - `trim_to_30sec.py` -> Trim audio to 30 sec (maximum audio length for classification)
- `cleaning_audio.py` -> Mark and remove audio files containing unmatched audio segments
- `parameter_check1.py` & `parameter_check1.py` -> Batch run of classifiers using combination of different hyperparameters to find better performing hyperparameters.
- `visualization.ipynb` -> Audio Visualizations
- `audio_classifier.ipynb` -> Audio Classifier


### Dataset
Dataset location: `/project-path/datasets/final_final`