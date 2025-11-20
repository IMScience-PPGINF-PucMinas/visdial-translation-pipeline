# VisDial Translation Evaluation

Manual evaluation tool for assessing the quality of VisDial dataset translations, based on the methodology from "Multilingual Visual Understanding: Extending Visual Dialog to Portuguese and Spanish Through Cross-Modal Adaptation" (CIARP 2025).

## Overview

This tool enables human evaluation of translated VisDial datasets (Portuguese and Spanish) using a standardized 3-point scale across three critical linguistic dimensions:

- **Fluency**: Grammatical correctness and naturalness of the translation
- **Adequacy**: Preservation of original meaning
- **Coherence**: Consistency with the dialog context

## Requirements

```bash
pip install torch transformers tqdm
```

## Usage

### Basic Evaluation

Evaluate 100 Portuguese translations from the validation set:

```bash
python evaluate_translations.py --language pt --n-samples 100
```

Evaluate 50 Spanish translations:

```bash
python evaluate_translations.py --language es --n-samples 50
```

### Advanced Options

```bash
python evaluate_translations.py \
  --language pt \
  --n-samples 100 \
  --input-dir ./translated \
  --output evaluation_results.txt \
  --split val \
  --seed 42
```

### Command-Line Arguments

- `--language`, `-l`: Target language code (required)
  - `pt`: Brazilian Portuguese
  - `es`: Spanish

- `--n-samples`, `-n`: Number of samples to evaluate (default: 100)

- `--input-dir`, `-i`: Directory containing VisDial JSON files (default: current directory)

- `--output`, `-o`: Output file for results (default: auto-generated with timestamp)

- `--split`, `-s`: Dataset split to evaluate (default: val)
  - `train`: Training set
  - `val`: Validation set
  - `test`: Test set

- `--seed`: Random seed for reproducibility (default: 42)

## File Structure

Your directory should contain:

```
.
├── visdial_1.0_val.json          # Original English dataset
├── visdial_1.0_val_PT.json       # Portuguese translation
├── visdial_1.0_val_ES.json       # Spanish translation
└── evaluate_translations.py       # This script
```

## Evaluation Process

1. The script randomly samples N examples (questions, answers, and captions) from the dataset
2. For each sample, it displays:
   - The original English text
   - The translated text
   - The type (Question, Answer, or Caption)
3. You rate each translation on a 1-3 scale for:
   - **Fluency** (1=Poor, 2=Good, 3=Excellent)
   - **Adequacy** (1=Poor, 2=Good, 3=Excellent)
   - **Coherence** (1=Poor, 2=Good, 3=Excellent)
4. Results are saved to a text file with:
   - Summary statistics (average scores)
   - Scores by type (Question/Answer/Caption)
   - Detailed individual evaluations

## Output Format

The evaluation results file includes:

```
================================================================================
VisDial Translation Evaluation Results - Portuguese
Date: 2025-01-12 14:30:00
Total Samples: 100
================================================================================

SUMMARY STATISTICS
--------------------------------------------------------------------------------
Average Fluency:   2.84 / 3.0
Average Adequacy:  2.89 / 3.0
Average Coherence: 2.86 / 3.0

SCORES BY TYPE
--------------------------------------------------------------------------------

Question (n=33):
  Fluency:   2.85
  Adequacy:  2.91
  Coherence: 2.88

Answer (n=34):
  Fluency:   2.82
  Adequacy:  2.88
  Coherence: 2.85

Caption (n=33):
  Fluency:   2.85
  Adequacy:  2.88
  Coherence: 2.85

================================================================================
DETAILED EVALUATION RESULTS
================================================================================

SAMPLE 1
--------------------------------------------------------------------------------
Type: Question

English (Original):
  Are there people in the room?

Portuguese (Translation):
  Há pessoas na sala?

Scores:
  Fluency:   3/3
  Adequacy:  3/3
  Coherence: 3/3

================================================================================
...
```

## Tips for Evaluation

1. **Be Consistent**: Use the same standards throughout the evaluation
2. **Consider Context**: Evaluate coherence in relation to the full dialog
3. **Natural Language**: Fluency should reflect how a native speaker would phrase it
4. **Meaning Preservation**: Adequacy focuses on semantic equivalence, not word-for-word translation
5. **Take Breaks**: For large evaluations, you can interrupt (Ctrl+C) and results will be saved

## Interrupt and Resume

- Press `Ctrl+C` at any time to stop evaluation
- All completed evaluations will be saved
- To continue later, run the script again with a different seed or modify completed samples

## Reference

This evaluation methodology is based on:

Milena Menezes Adão, Silvio Jamil F. Guimarães, and Zenilton K. G. Patrocínio Jr. "Multilingual Visual Understanding: Extending Visual Dialog to Portuguese and Spanish Through Cross-Modal Adaptation." In Proceedings of CIARP 2025.

## Expected Results

Based on the paper, high-quality translations should achieve:

**Portuguese (VisDial-PT):**
- Fluency: ~2.84 / 3.0
- Adequacy: ~2.89 / 3.0
- Coherence: ~2.86 / 3.0

**Spanish (VisDial-ES):**
- Fluency: ~2.91 / 3.0
- Adequacy: ~2.93 / 3.0
- Coherence: ~2.88 / 3.0

## License

This evaluation tool is provided as part of the VisDial multilingual dataset creation pipeline.