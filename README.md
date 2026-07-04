# indic deepfake speech detection

binary classifier to detect genuine human speech vs tts-synthesized audio across 16 indian languages.

## how it works

when a machine generates speech, it leaves behind subtle fingerprints: pitch that is a little too steady, transitions that are a little too smooth, background texture that is a little too clean. humans rarely notice them, but they are measurable.

this project catches those fingerprints in two complementary ways:

1. **a big pretrained "ear"**: a wav2vec2 speech model that has already listened to thousands of hours of speech. we lightly fine-tune it (lora) to tell real from fake, and use its internal representations as rich descriptions of each clip.
2. **a set of hand-picked acoustic measurements**: pitch stability, phase continuity, harmonic balance, loudness variation and so on - the exact things tts systems tend to get wrong.

a gradient-boosted tree model (xgboost) then combines both views, plus simple text metadata, into one probability per clip: how likely is this audio machine-generated?

to keep the evaluation honest, the training data is split into 5 folds and each clip is only ever described by a model that never saw it during training (out-of-fold). the 5 fold models also double as a free ensemble at prediction time, and the final score blends the tree model with the neural network's own opinion - whichever mix works best on held-out data. a suspiciously perfect score is treated as a bug report, not a result - see [results](#results) below for what that turned up.

## project structure

* [notebooks/model_training.ipynb](notebooks/model_training.ipynb) - main training pipeline. caches waveforms, extracts handcrafted features, fine-tunes `ai4bharat/indicwav2vec-hindi` with lora across 5 group-aware out-of-fold splits, trains an xgboost classifier on the fused features, blends predictions, exports the best fold to onnx, and scores generalization on a held-out synthesis engine.
* [scripts/validate_on_extra_dataset.py](scripts/validate_on_extra_dataset.py) - standalone local re-run of the out-of-distribution check against downloaded kaggle artifacts, without needing the full notebook or GPU.
* [helpers/](helpers) - helper utilities:
  * [eda.ipynb](helpers/eda.ipynb) - exploratory analysis of dataset class balance, languages, transcription text, and acoustic features.
  * [dataset_generation.ipynb](helpers/dataset_generation.ipynb) - dataset generator using `kenpath/svara-tts-v1` and snac vocoders to create synthetic audio clips for class balancing.
* [requirements.txt](requirements.txt) - dependency list.
* [.gitignore](.gitignore) - ignores local data, checkouts, and system files.

## pipeline design

* **waveform cache**: every clip is decoded, resampled to 16 khz, normalized, and cropped/padded to 5 s exactly once, then stored as int16 pcm in a disk-backed memmap. every later stage reads from this cache instead of re-decoding audio - the single biggest speed win in the pipeline.
* **handcrafted features (36-d)**: pitch mean/std, spectral flux, phase coherence, harmonics-to-noise ratio, rms mean/std, spectral flatness mean/std, duration, and mean/std of 13 mfccs - extracted in parallel across all cpu cores.
* **duplicate & leakage checks**: every clip is hashed by exact audio content and paired by `(language, text)`; linked clips are grouped so a fold split can never separate a matched real/fake pair, and the same hashes rule out literal train/test overlap. a **shortcut audit** also checks whether any single handcrafted feature alone predicts the label - none does (best is 0.62 auc), so the signal isn't a trivial artifact like duration or loudness.
* **model & embedding fusion**: fine-tunes `ai4bharat/indicwav2vec-hindi` (hidden size 1024, 24 layers) with lora on all attention projections (~3.1m / 1% trainable, fp16 mixed precision). representations from a middle layer (local acoustic/vocoder artifacts) and the final layer (semantic structure) are pooled with mean + std into a 4096-d embedding. dimensions are read from the model config, so swapping the backbone is a one-line change.
* **5-fold out-of-fold (oof)**: five lora fold models split with `StratifiedGroupKFold` on the duplicate groups above; each embeds only the held-out fold it never saw, so every training embedding is leak-free. the five fold models double as a free test-time ensemble - no separate final model is trained.
* **classifier + blend**: xgboost trains on [oof embeddings | handcrafted | metadata] with early stopping and class-imbalance weighting; the final prediction blends its probability with the fold ensemble's own output, whichever mix wins on the validation split.
* **onnx export**: the fold with the best held-out auc is merged and exported for deployment without pytorch or peft.

## results

leak-free 5-fold out-of-fold evaluation on the main challenge dataset (31k clips, 16 languages), vs. the same trained ensemble scored on a held-out dataset built with a different tts engine and a different genuine-speech source it never trained on:

| metric | main dataset (in-distribution) | unseen tts engine (out-of-distribution) |
| --- | --- | --- |
| roc auc | 0.9995 | 0.826 |
| eer | 0.84% | 24.0% |
| accuracy @ 0.5 | 96.7% | 79.6% (81.3% genuine / 65.7% synthetic) |

the in-distribution number is real, but narrow: it reflects the model fingerprinting the specific tts systems used to build this one dataset, not synthetic speech in general. scored against audio from a synthesis engine and a recording pipeline it has never seen, performance drops sharply in both directions - more genuine clips misread as fake, more fake clips missed entirely. that gap is the honest measure of how this model generalizes, and it's the number worth trusting over the flashier in-distribution score.

full per-language breakdown, per-fold aucs, onnx deployment numbers (1.26 gb, <0.01% output difference vs pytorch, ~1.3s/clip on cpu), and the out-of-distribution report are written to `evaluation_report.json`, `per_language_metrics.csv`, `extra_dataset_ood_report.json`, and a single readable `RUN_REPORT.md` at the end of the run.

## dataset info

* **main dataset**: SherryT997/IndicTTS-Deepfake-Challenge-Data (~31k train / ~2.6k test samples, CC-BY-4.0), used for both training and the leaderboard submission.
* **extra dataset**: springlab's open genuine-speech corpuses (1,200 recordings) plus synthetic clips from `kenpath/svara-tts-v1` + snac vocoders (140 clips). **not used for training** - a different recording pipeline for the genuine half and a different tts engine for the synthetic half made it a source/channel shortcut risk when merged in, so it's kept aside and used only as the out-of-distribution check above.
* **preprocessing**: all clips resampled to 16khz (16-bit pcm), peak-normalized, padded/cropped to 5 seconds (80,000 samples).

## running on kaggle

1. mount both inputs in kaggle settings: `iveeaten3223times/multilingual-indian-speech-data` (training) and `adhithyasash1/indic-deepfake-challenge-extra-dataset` (used only by the out-of-distribution check at the end).
2. upload and run the training notebook with gpu and internet enabled.
3. for a quick smoke check first, set `TRAIN_LIMIT = 500`, `TEST_LIMIT = 100`, `N_EPOCHS = 1` in the config cell, then reset to `None` / `3` for the full run.
4. progress is checkpointed after every fold, so a restarted session resumes where it left off; stale caches are detected and discarded automatically.
5. outputs worth downloading: `RUN_REPORT.md`, `evaluation_report.json`, `extra_dataset_ood_report.json`, `per_language_metrics.csv`, `shortcut_audit.csv`, `submission.csv`, and `deepfake_detector.onnx`. the rest (waveform caches, stacked feature matrices) are multi-gb intermediates with no value once the run finishes.
