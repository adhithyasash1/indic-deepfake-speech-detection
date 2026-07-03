# indic deepfake speech detection

binary classifier to detect genuine human speech vs tts-synthesized audio across 16 indian languages.

## how it works

when a machine generates speech, it leaves behind subtle fingerprints: pitch that is a little too steady, transitions that are a little too smooth, background texture that is a little too clean. humans rarely notice them, but they are measurable.

this project catches those fingerprints in two complementary ways:

1. **a big pretrained "ear"**: a wav2vec2 speech model that has already listened to thousands of hours of speech. we lightly fine-tune it (lora) to tell real from fake, and use its internal representations as rich descriptions of each clip.
2. **a set of hand-picked acoustic measurements**: pitch stability, phase continuity, harmonic balance, loudness variation and so on - the exact things tts systems tend to get wrong.

a gradient-boosted tree model (xgboost) then combines both views, plus simple text metadata, into one probability per clip: how likely is this audio machine-generated?

to keep the evaluation honest, the training data is split into 5 folds and each clip is only ever described by a model that never saw it during training (out-of-fold). the 5 fold models also double as a free ensemble at prediction time, and the final score blends the tree model with the neural network's own opinion - whichever mix works best on held-out data.

## project structure

* [notebooks/model_training.ipynb](notebooks/model_training.ipynb) - main training pipeline. caches waveforms, extracts handcrafted features, fine-tunes `ai4bharat/indicwav2vec-hindi` with lora across 5 out-of-fold splits, trains an xgboost classifier on the fused features, blends predictions, and exports the best fold to onnx.
* [helpers/](helpers) - helper utilities including:
  * [eda.ipynb](helpers/eda.ipynb) - exploratory analysis of dataset class balance, languages, transcription text, and acoustic features.
  * [dataset_generation.ipynb](helpers/dataset_generation.ipynb) - dataset generator using `kenpath/svara-tts-v1` and snac vocoders to create synthetic audio clips for class balancing.
* [requirements.txt](requirements.txt) - dependency list.
* [.gitignore](.gitignore) - ignores local data, checkouts, and system files.

## pipeline design

* **waveform cache**: every clip is decoded, resampled to 16 khz, normalized, and cropped/padded to 5 s exactly once, then stored as int16 pcm in a disk-backed memmap. all later stages (feature extraction, every training epoch of every fold) read from this cache instead of re-decoding audio, which is the single biggest speed win in the pipeline.
* **handcrafted features (36-d)**: statistical audio cues targeting synthetic tells - pitch mean/std, spectral flux, phase coherence, harmonics-to-noise ratio (hnr), rms mean/std, spectral flatness mean/std, signal duration, and mean/std of 13 mfccs. extracted in parallel across all cpu cores.
* **text metadata features**: one-hot encoded languages plus character length, word count, and unique-to-total character ratio of the transcription.
* **model & embedding fusion**: fine-tunes `ai4bharat/indicwav2vec-hindi` (a large checkpoint: hidden size 1024, 24 layers) with lora on all attention projections (~3m trainable parameters, fp16 mixed precision, onecycle lr schedule). representations from the middle layer (local acoustic/vocoder artifacts) and the final layer (semantic structure) are concatenated and pooled with mean + std over time into a 4096-d embedding. all dimensions are derived from the model config at load time, so swapping the backbone (e.g. `facebook/wav2vec2-xls-r-300m`) is a one-line change.
* **5-fold out-of-fold (oof)**: five lora fold models; each embeds only the held-out fold it never saw, producing 100% leak-free training features for the classifier. each fold model also runs over the test set, so the five models form a free test-time ensemble - no separate final model is trained.
* **classifier + blend**: xgboost is trained on [oof embeddings | handcrafted | metadata] with early stopping on a 15% stratified validation split and scale_pos_weight for class imbalance. the final prediction blends the xgboost probability with the fold ensemble's own sigmoid output; the blend weight is selected by validation auc.
* **onnx export**: the fold with the best held-out auc gets its lora adapter merged into the base model and exported to onnx together with its trained head, for deployment without pytorch or peft.

## dataset info

* **main dataset**: SherryT997/IndicTTS-Deepfake-Challenge-Data (~31k train / ~2.6k test samples, CC-BY-4.0) featuring speech and deepfakes across 16 indian languages.
* **extra dataset**: custom balanced dataset created to prevent class imbalance, merging:
  * **real human speech**: 1,200 recordings from springlab's open-source language corpuses on hugging face (covering languages like assamese, bengali, gujarati, hindi, kannada, malayalam, marathi, manipuri, odia, tamil, telugu).
  * **synthetic deepfakes**: 140 synthetic clips generated via svara casual text-to-speech (`kenpath/svara-tts-v1`) with snac vocoders, generated with alternate speaker genders and emotion tags (`<clear>`, `<happy>`, `<sad>`, `<fear>`, and `<anger>`).
* **preprocessing**: all audio clips are resampled to 16khz (16-bit pcm), peak-normalized, and padded or cropped to a maximum length of 5 seconds (80,000 samples).

## running on kaggle

1. mount these inputs in kaggle settings:
   * main dataset: `iveeaten3223times/multilingual-indian-speech-data`
   * extra dataset: `adhithyasash1/indic-deepfake-challenge-extra-dataset`
2. upload and run the training notebook with gpu and internet enabled.
3. run a quick smoke check first by setting `TRAIN_LIMIT = 500`, `TEST_LIMIT = 100`, and `N_EPOCHS = 1` in the config cell, then reset them to `None` / `3` for the full run.
4. progress is checkpointed after every fold (embeddings, adapters, aucs), so a restarted session resumes where it left off; stale caches from crashed runs are detected and discarded automatically.
5. outputs: `submission.csv`, `metrics.json`, per-fold lora adapters, and `deepfake_detector.onnx`.
