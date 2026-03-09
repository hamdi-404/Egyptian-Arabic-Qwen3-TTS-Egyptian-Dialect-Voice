# 🗣️ Egyptian Arabic Qwen3-TTS — Egyptian Dialect Voice

<p align="center">
  <img src="https://img.shields.io/badge/Language-Arabic%20(Egyptian)-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Base%20Model-Qwen3--TTS%201.7B-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-RTX%203060%2012GB-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?style=for-the-badge"/>
</p>

A fine-tuned **Qwen3-TTS 1.7B** model for generating **Egyptian Arabic speech** — a dialect that is rich, deeply rooted in Egyptian oral tradition, and almost entirely absent from modern TTS systems.

This project was trained from scratch on a **consumer GPU (RTX 3060 12GB)** with no cloud compute, proving that dialect-specific Speech AI is achievable without massive infrastructure.

> **Author:** Hamdi Mohamed — AI Engineer  
> **HuggingFace Model:** [itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base)

---

## 📋 Table of Contents

- [Why Egyptian Arabic?](#-why-egyptian-arabic)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Audio Processing Pipeline](#-audio-processing-pipeline)
- [Training Setup](#-training-setup)
- [Issues & Solutions](#-issues--solutions)
- [Installation](#-installation)
- [Inference](#-inference)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Why Egyptian Arabic?

The **Egyptian dialect** is one of the most widely spoken Arabic dialects in Egypt — yet it is almost completely unrepresented in modern AI speech systems. Most existing TTS models either:

- Produce a generic Modern Standard Arabic (MSA) voice
- Approximate the Cairo (Cairene) dialect at best
- Sound entirely foreign when given Egyptian text

The oral heritage of Egypt is enormous. Works like the epic **Sirat Bani Hilal**, as narrated by poets like **Abd al-Rahman al-Abnudi**, represent centuries of spoken Egyptian tradition that deserves to be preserved and synthesized accurately.

This project was built to give the Egyptian dialect its own TTS voice.

---

## 🧠 Model Architecture

The base model is **Qwen3-TTS 1.7B**, a multilingual transformer-based text-to-speech system.

```
Text Input
    │
    ▼
┌─────────────────┐
│  Text Tokenizer │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Talker Model   │  ← Transformer (1.7B params)
│  (Transformer)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Speaker Encoder │  ← egyptian_speaker embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Codec Decoder  │  ← 12Hz speech codec
└────────┬────────┘
         │
         ▼
   Audio Waveform
```

### Key Properties

| Property | Value |
|---|---|
| **Parameters** | 1.7B |
| **Speech Codec** | 12Hz |
| **Architecture** | Transformer |
| **Voice Control** | Speaker Embeddings |
| **Languages** | Multilingual (fine-tuned for Egyptian Arabic) |
| **Training Precision** | bf16 |
| **Attention** | Flash Attention 2 |

---

## 📦 Dataset

### Source

The training data was collected primarily from **Abd al-Rahman al-Abnudi** reciting **Sirat Bani Hilal** — one of the greatest examples of Egyptian oral poetry and storytelling. This made the model naturally learn authentic Egyptian dialect prosody, rhythm, and pronunciation.

### Format

The dataset uses a JSONL format required by the Qwen3-TTS training pipeline:

```json
{
  "audio": "data_24k/wavs/train_sample_17746.wav",
  "text": "تدريجيا بقى أيامها في أعلى من المعدل",
  "ref_audio": "data_24k/ref/ref_audio.wav"
}
```

| Field | Description |
|---|---|
| `audio` | Path to the speech waveform (.wav) |
| `text` | Arabic transcription of the audio |
| `ref_audio` | Reference speaker audio for speaker encoder |

### Dataset Iterations

The dataset went through multiple stages before reaching the final version:

| Stage | Size | Result |
|---|---|---|
| Stage 1 | ~7 hours | Model learned Egyptian but with inconsistencies |
| Stage 2 | ~40 hours | Expanded — still had audio quality artifacts |
| Stage 3 | ~20 hours clean ✅ | Best results — quality over quantity |
| **Final** | **~90 hours** | Current training set after full curation |

> **Key lesson:** Cleaning and filtering the data had more impact than increasing its size. 20 hours of clean speech outperformed 40 hours of mixed-quality data.

---

## 🔊 Audio Processing Pipeline

Before training, all audio was preprocessed through the following steps:

### Step 1 — Resampling

All audio files were resampled to **24kHz** to match the Qwen3-TTS codec requirements:

```bash
ffmpeg -i input.wav -ar 24000 output.wav
```

### Step 2 — Speech Filtering

Music, noise, and non-speech segments were removed using **inaSpeechSegmenter**:

```python
from inaSpeechSegmenter import Segmenter

seg = Segmenter()
segmentation = seg("audio_file.wav")

# Keep only segments labeled as 'male' or 'female' (speech)
speech_segments = [(start, end) for label, start, end in segmentation 
                   if label in ['male', 'female']]
```

### Step 3 — Speech Tokenization

Audio was converted to speech tokens using the Qwen3-TTS tokenizer:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

The output `train_with_codes.jsonl` contains:
- Speech codec tokens
- Text tokens
- Speaker metadata

---

## ⚙️ Training Setup

### Hardware

| Component | Spec |
|---|---|
| **GPU** | NVIDIA RTX 3060 12GB VRAM |
| **Training Type** | Local (no cloud) |
| **Framework** | PyTorch + HuggingFace Transformers |

> The entire model was fine-tuned on a consumer-grade GPU. Flash Attention 2 and bf16 precision were critical to fit the 1.7B model within 12GB VRAM.

### Hyperparameters

| Parameter | Value |
|---|---|
| **Learning Rate** | `2e-6` |
| **Batch Size** | `2` |
| **Gradient Accumulation Steps** | `8` |
| **Effective Batch Size** | `16` |
| **Epochs** | `3` |
| **Precision** | `bf16` |
| **Attention** | Flash Attention 2 |
| **Optimizer** | AdamW |

### Training Command

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_model \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 3 \
  --speaker_name egyptian_speaker
```

### Training Log (Sample)

```
Epoch 1/3 | Loss: 9.41
Epoch 2/3 | Loss: 8.23
Epoch 3/3 | Loss: 7.80
```

Loss decreased steadily across all epochs, indicating successful learning of Egyptian speech patterns.

---

## 🐛 Issues & Solutions

Several significant issues were encountered and resolved during development:

### 1. NaN Loss During Training

**Problem:**
```
RuntimeError: Loss became NaN during training
```

**Cause:** Learning rate too high for the model's initial state.

**Fix:**
```bash
# Reduce learning rate and add NaN guards
python sft_12hz.py \
  --lr 2e-6 \          # reduced from higher value
  --skip_nan \         # skip batches that produce NaN
  --nan_backoff        # reduce LR automatically on NaN
```

---

### 2. SoX Not Found

**Problem:**
```
FileNotFoundError: SoX could not be found
```

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install sox

# Add to PATH if needed
export PATH=$PATH:/usr/bin/sox
```

---

### 3. Transformers Version Mismatch

**Problem:**
```
ImportError: cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers'
```

**Fix:**
```bash
pip install transformers==4.40.0
```

---

### 4. Incorrect Model Export

**Problem:** The model was initially exported without the correct tokenizer files and speaker config, causing inference to fail silently.

**Symptoms:**
- Missing `tokenizer_config.json`
- Missing `generation_config.json`
- Speaker config mismatch (speaker ID not found)

**Fix:** Manually reconstruct the export directory:

```bash
# Copy tokenizer files from base model
cp Qwen/Qwen3-TTS-12Hz-1.7B-Base/tokenizer_config.json output_model/
cp Qwen/Qwen3-TTS-12Hz-1.7B-Base/generation_config.json output_model/

# Patch config.json to include speaker mapping
# config.json should include:
# "tts_model_type": "custom_voice"
# "spk_id": {"egyptian_speaker": 3000}
```

---

## 🛠️ Installation

> ⚠️ **Note:** The model weights are hosted exclusively on **HuggingFace**. This GitHub repository contains only documentation. To use the model, follow the steps below.

### Step 1 — Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0
pip install soundfile
pip install qwen-tts
sudo apt-get install sox
```

### Step 2 — Load the Model from HuggingFace

The model will be downloaded automatically from HuggingFace on first use:

```python
from qwen_tts import Qwen3TTSModel
import torch

tts = Qwen3TTSModel.from_pretrained(
    "itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base",  
    device_map={"": 0},
    torch_dtype=torch.float16,
)
```

> 🔗 Model page: [huggingface.co/itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base)

---

## 🚀 Inference

### Basic Usage

```python
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

# Load the model
tts = Qwen3TTSModel.from_pretrained(
    "itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base",
    device_map={"": 0},
    torch_dtype=torch.float16,
)

# Generate Egyptian Arabic speech
wavs, sr = tts.generate_custom_voice(
    text="إزيك يا صاحبي عامل إيه النهاردة",
    speaker="egyptian_speaker",
    language="auto",
)

sf.write("output.wav", wavs[0], sr)
print(f"Audio saved — sample rate: {sr}Hz")
```

### With Sampling Parameters

```python
wavs, sr = tts.generate_custom_voice(
    text="النهاردة الجو جميل جدا في الصعيد",
    speaker="egyptian_speaker",
    language="auto",
    temperature=0.8,   # Lower = more consistent, Higher = more varied
    top_p=0.9,         # Nucleus sampling threshold
)

sf.write("output.wav", wavs[0], sr)
```

### Batch Inference

```python
texts = [
    "إزيك يا صاحبي عامل إيه النهاردة",
    "النهاردة الجو جميل جدا في الصعيد",
    "أنا سعيد إننا قدرنا نبني موديل بيتكلم صعيدي",
]

for i, text in enumerate(texts):
    wavs, sr = tts.generate_custom_voice(
        text=text,
        speaker="egyptian_speaker",
        language="auto",
    )
    sf.write(f"sample_{i+1}.wav", wavs[0], sr)
    print(f"Generated: sample_{i+1}.wav")
```

---

## 📊 Results

### What Worked

| Aspect | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| **Dialect** | Generic/Foreign accent | Authentic Egyptian |
| **Prosody** | Flat and robotic | Natural conversational rhythm |
| **Pronunciation** | Inconsistent | Clear and consistent |
| **Oral tradition feel** | None | Strong (Al-Abnudi style) |

### Model Configuration

```json
{
  "tts_model_type": "custom_voice",
  "spk_id": {
    "egyptian_speaker": 3000
  }
}
```

---

## 📁 Where to Find Everything

This GitHub repository contains **documentation only**.  
All model files, weights, and audio samples are hosted on HuggingFace:

| Resource | Location |
|---|---|
| **Model weights** | [HuggingFace — itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base) |
| **Audio samples** | Available on the HuggingFace model page |
| **Model config** | Included in the HuggingFace repo (`config.json`) |
| **Tokenizer files** | Included in the HuggingFace repo |
| **Documentation** | This GitHub README |

### Training Pipeline (Reference)

The training pipeline used the following structure locally:

```
training_workspace/
│
├── data_24k/
│   ├── wavs/                   # Resampled 24kHz audio
│   └── ref/
│       └── ref_audio.wav       # Reference speaker audio
│
├── train_raw.jsonl             # Raw dataset
├── train_with_codes.jsonl      # Tokenized dataset (after prepare_data.py)
│
├── prepare_data.py             # Speech tokenization script
└── sft_12hz.py                 # Fine-tuning script
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Single speaker** — only one Egyptian voice (Al-Abnudi style) is available
- **Sub-dialect variation** — the model reflects one specific Egyptian speaker style; other regional Egyptian variations are not covered
- **Uncommon vocabulary** — rare or modern words may be mispronounced
- **Emotional range** — the model produces a single speaking style (storytelling/narrative)

### Planned Improvements

- [ ] **Multi-speaker Egyptian dataset** — diverse egyptian voices
- [ ] **Larger dataset** — scale to 200+ hours
- [ ] **Emotion control** — happy, sad, formal, energetic via instruction prompts
- [ ] **Speaking rate control** — slow/fast speech
- [ ] **Other dialects** — Cairene, Levantine, Gulf, Maghrebi
- [ ] **Evaluation metrics** — MOS scores, WER on Egyptian text

---

## 📄 Citation

If you use this model or dataset in your research, please cite:

```bibtex
@misc{hamdi2025egyptianqwen3tts,
  title     = {Egyptian Arabic Qwen3-TTS: Fine-tuning Large TTS Models for Regional Arabic Dialects},
  author    = {Hamdi Mohamed},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base}
}
```

---

## 📜 License

This project is licensed under the **Apache 2.0 License** — consistent with the base Qwen3-TTS model license.

See [LICENSE](LICENSE) for full details.

---

## 🙏 Acknowledgements

- **Qwen Team (Alibaba)** for releasing Qwen3-TTS as an open base model
- **Abd al-Rahman al-Abnudi** — whose voice and storytelling tradition inspired this project and provided the core training data

---

<p align="center">
  Built with ❤️ for the Arabic-speaking world 🌍
</p>
