# indic deepfake speech detection

binary classifier to detect genuine human speech vs tts-synthesized audio across 16 indian languages.

## project structure

* [model_training.ipynb](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/notebooks/model_training.ipynb) - main training pipeline. extracts handcrafted features (pitch, mfccs, spectral flux), fine-tunes `ai4bharat/indicwav2vec` using lora, extracts 5-fold out-of-fold fused embeddings (layer 6 + 12), and trains an xgboost classifier.
* [helpers/](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/helpers) - helper utilities including:
  * [eda.ipynb](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/helpers/eda.ipynb) - exploratory analysis of dataset class balance, languages, transcription text, and acoustic features.
  * [dataset_generation.ipynb](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/helpers/dataset_generation.ipynb) - dataset generator using `kenpath/svara-tts-v1` and snac vocoders to create synthetic audio clips for class balancing.
* [requirements.txt](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/requirements.txt) - dependency list.
* [.gitignore](file:///Users/adhithyasash1/Downloads/indic%20audio/indic-deepfake/indic-deepfake-speech-detection/.gitignore) - ignores local data, checkouts, and system files.

## pipeline design

* **handcrafted features (33-d)**: extracts statistical audio cues to identify synthetic tells. includes pitch mean/std, spectral flux, phase coherence, harmonics-to-noise ratio (hnr), rms variability, signal duration, and mean/std of 13 mfccs.
* **text metadata features**: one-hot encodes target languages and extracts character length, word counts, and unique-to-total character ratios to capture transcription patterns.
* **model & embedding fusion**: fine-tunes `ai4bharat/indicwav2vec` using lora (~1m parameters). extracts and concatenates representation vectors from both the acoustic layer (layer 6, capturing vocoder artifacts and phase discontinuities) and the semantic layer (layer 12, capturing phone transitions). mean + std pooling are applied to extract a 3072-d embedding.
* **5-fold out-of-fold (oof)**: trains five separate lora fold models. each model embeds the held-out fold it never saw during training. this produces 100% leak-free training embeddings for the classifier, preventing representation collapse.
* **final classifier**: an xgboost model trained on the combined handcrafted, metadata, and fused wav2vec embeddings, utilizing early stopping on a 15% stratified validation split, and scale_pos_weight to manage class imbalance.

## dataset info

* **main dataset**: SherryT997/IndicTTS-Deepfake-Challenge-Data (~31k train / ~2.6k test samples, CC-BY-4.0) featuring speech and deepfakes across 16 indian languages.
* **extra dataset**: custom balanced dataset created to prevent class imbalance, merging:
  * **real human speech**: 1,200 recordings from springlab's open-source language corpuses on hugging face (covering languages like assamese, bengali, gujarati, hindi, kannada, malayalam, marathi, manipuri, odia, tamil, telugu).
  * **synthetic deepfakes**: 140 synthetic clips generated via svara casual text-to-speech (`kenpath/svara-tts-v1`) with snac vocoders, generated with alternate speaker genders and emotion tags (`<clear>`, `<happy>`, `<sad>`, `<fear>`, and `<anger>`).
* **preprocessing**: all audio clips are resampled to 16khz (16-bit pcm wav), normalized, and padded or cropped to a maximum length of 5 seconds (80,000 samples).

## running on kaggle

1. mount these inputs in kaggle settings:
   * main dataset: `iveeaten3223times/multilingual-indian-speech-data`
   * extra dataset: `adhithyasash1/indic-deepfake-challenge-extra-dataset`
2. upload and run the training notebook with gpu and internet enabled.
3. run a quick smoke check by setting `TRAIN_LIMIT = 200` and `N_EPOCHS = 1`.
