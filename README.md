# VisDial Multilingual Translation Pipeline

Complete pipeline for translating VisDial datasets to Portuguese, Spanish or other language, with human evaluation tools.

**Based on:** "Multilingual Visual Understanding: Extending Visual Dialog to Portuguese and Spanish Through Cross-Modal Adaptation" (CIARP 2025)

## Overview

This repository provides:

1. **Translation Pipeline** (`translate_visdial.py`) - Automatic translation of VisDial datasets using MarianMT models
2. **Evaluation Tool** (`evaluate_translations.py`) - Manual evaluation of translation quality
3. **Demo Script** (`demo_evaluation.py`) - Example of the evaluation process

## Quick Start



### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Translate Dataset

Translate VisDial to Portuguese:

```bash
python translate_visdial.py \
  --language pt \
  --input-dir ./original \
  --output-dir ./translated \
  --splits val
```

Translate to Spanish:

```bash
python translate_visdial.py \
  --language es \
  --input-dir ./original \
  --output-dir ./translated \
  --splits val
```

### 3. Evaluate Translations

Evaluate 100 random samples:

```bash
python evaluate_translations.py \
  --language pt \
  --n-samples 100 \
  --input-dir ./translated
```

## Pipeline Components

### Translation Script (`translate_visdial.py`)

Translates VisDial datasets using state-of-the-art neural machine translation:

**Features:**
- Batch processing for efficiency
- GPU acceleration support
- Translation caching to avoid duplicates
- Preserves original dataset structure
- Progress tracking with tqdm

**Models Used:**
- Portuguese: `Helsinki-NLP/opus-mt-tc-big-en-pt`
- Spanish: `Helsinki-NLP/opus-mt-en-es`

**Usage:**

```bash
python translate_visdial.py --help

# Translate validation set to Portuguese
python translate_visdial.py -l pt -s val

# Translate all splits to Spanish with custom settings
python translate_visdial.py -l es -s train val test -b 64 --cache-dir ./models

# Use CPU only
python translate_visdial.py -l pt --no-gpu
```

### Evaluation Script (`evaluate_translations.py`)

Manual evaluation tool following the paper's methodology:

**Evaluation Criteria (1-3 scale):**
- **Fluency**: Grammatical correctness and naturalness
- **Adequacy**: Preservation of original meaning
- **Coherence**: Consistency with dialog context

**Usage:**

```bash
python evaluate_translations.py --help

# Evaluate Portuguese translations
python evaluate_translations.py -l pt -n 100

# Evaluate Spanish with custom output
python evaluate_translations.py -l es -n 50 -o my_evaluation.txt

# Evaluate training set
python evaluate_translations.py -l pt -n 100 -s train
```

### Demo Script (`demo_evaluation.py`)

Shows examples of the evaluation process:

```bash
python demo_evaluation.py
```

## Directory Structure

```
.
├── translate_visdial.py          # Main translation script
├── evaluate_translations.py      # Evaluation tool
├── demo_evaluation.py            # Demo/example script
├── EVALUATION_README.md          # Detailed evaluation guide
├── original/                     # Original English datasets
│   ├── visdial_1.0_train.json
│   ├── visdial_1.0_val.json
│   └── visdial_1.0_test.json
└── translated/                   # Translated datasets
    ├── visdial_1.0_train_PT.json
    ├── visdial_1.0_train_ES.json
    ├── visdial_1.0_val_PT.json
    ├── visdial_1.0_val_ES.json
    ├── visdial_1.0_test_PT.json
    └── visdial_1.0_test_ES.json
```

## Dataset Statistics

Each translated dataset maintains the same structure as VisDial v1.0:

- **Training set**: ~120,000 images with dialogs
- **Validation set**: 2,000 images
- **Test set**: 8,000 images
- **Total QA pairs per language**: ~1.2 million

## Translation Quality

Based on the CIARP 2025 paper, expected evaluation scores:

### Portuguese (VisDial-PT)
- Fluency: 2.84 / 3.0
- Adequacy: 2.89 / 3.0
- Coherence: 2.86 / 3.0

### Spanish (VisDial-ES)
- Fluency: 2.91 / 3.0
- Adequacy: 2.93 / 3.0
- Coherence: 2.88 / 3.0

## Complete Workflow

### Step 1: Prepare Data

Download the original VisDial v1.0 dataset:
- Training: `visdial_1.0_train.json`
- Validation: `visdial_1.0_val.json`
- Test: `visdial_1.0_test.json`

### Step 2: Translate

```bash
# Portuguese
python translate_visdial.py \
  --language pt \
  --input-dir ./original \
  --output-dir ./translated \
  --splits train val test \
  --batch-size 32

# Spanish
python translate_visdial.py \
  --language es \
  --input-dir ./original \
  --output-dir ./translated \
  --splits train val test \
  --batch-size 32
```

### Step 3: Evaluate Quality

```bash
# Evaluate Portuguese validation set
python evaluate_translations.py \
  --language pt \
  --n-samples 100 \
  --input-dir ./translated \
  --split val \
  --output evaluation_pt_val.txt

# Evaluate Spanish validation set
python evaluate_translations.py \
  --language es \
  --n-samples 100 \
  --input-dir ./translated \
  --split val \
  --output evaluation_es_val.txt
```

### Step 4: Review Results

Check the generated evaluation files for:
- Summary statistics
- Scores by type (Question/Answer/Caption)
- Detailed individual evaluations

## Hardware Requirements

### Translation
- **GPU**: Recommended (CUDA-capable)
  - 8GB+ VRAM for batch size 32
  - Can use CPU with `--no-gpu` flag (slower)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB for models + dataset size

### Evaluation
- **CPU**: Any modern processor
- **RAM**: 4GB+
- No GPU required

## Performance

### Translation Speed (GPU)
- Validation set (~10,000 dialogs): ~15-30 minutes
- Training set (~120,000 dialogs): ~3-5 hours
- Batch size 32 with NVIDIA RTX 3090

### Evaluation Time
- 100 samples: ~20-30 minutes (manual evaluation)
- 1000 samples: ~3-4 hours (manual evaluation)

## Extending to Other Languages

The pipeline can be extended to other languages by:

1. Finding appropriate MarianMT models on HuggingFace
2. Adding the model to the `MODELS` dictionary in `translate_visdial.py`
3. Adding the language code to `LANGUAGES` in `evaluate_translations.py`

Example for French:
```python
MODELS = {
    "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-en-fr",  # Add French
}
```

## Citation

If you use this pipeline or the datasets, please cite:

```bibtex
@inproceedings{adao2025multilingual,
  title={Multilingual Visual Understanding: Extending Visual Dialog to Portuguese and Spanish Through Cross-Modal Adaptation},
  author={Ad{\~a}o, Milena Menezes and Guimar{\~a}es, Silvio Jamil F. and Patroc{\'\i}nio Jr, Zenilton K. G.},
  booktitle={Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications (CIARP)},
  year={2025}
}
```


## Authors

- Milena Menezes Adão
- Silvio Jamil F. Guimarães
- Zenilton K. G. Patrocínio Jr.

Pontifícia Universidade Católica de Minas Gerais – Belo Horizonte, Brazil

## Acknowledgments

This work is based on the original VisDial dataset:
- Das, A., et al. "Visual Dialog." CVPR 2017.

Translation models from:
- Helsinki-NLP OPUS-MT project

## Support

For issues, questions, or contributions, please [open an issue/contact information].

## Related Resources

- Original VisDial: https://visualdialog.org/
- MarianMT Models: https://huggingface.co/Helsinki-NLP
