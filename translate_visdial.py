#!/usr/bin/env python3
"""
VisDial Translation Pipeline
Translates VisDial datasets to Portuguese or Spanish
Based on the paper: "Multilingual Visual Understanding" (CIARP 2025)
"""

import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


class VisDialTranslator:
    """Translates VisDial datasets to target languages using MarianMT models."""

    # Model mapping for supported languages
    MODELS = {
        "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",  # Portuguese
        "es": "Helsinki-NLP/opus-mt-en-es",  # Spanish
    }

    def __init__(
        self,
        target_language: str = "es",
        batch_size: int = 32,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the translator.

        Args:
            target_language: Target language code ('pt' or 'es')
            batch_size: Batch size for translation
            use_gpu: Whether to use GPU if available
            cache_dir: Directory to cache models
        """
        if target_language not in self.MODELS:
            raise ValueError(
                f"Language {target_language} not supported. Choose from: {list(self.MODELS.keys())}"
            )

        self.target_language = target_language
        self.batch_size = batch_size
        self.model_name = self.MODELS[target_language]

        # Setup device
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )

        # Load model and tokenizer
        print(f"Loading model: {self.model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        self.model = MarianMTModel.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = self.model.to(self.device)

        # Translation cache for efficiency
        self.translation_cache = {}

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of texts to translate

        Returns:
            List of translated texts
        """
        # Tokenize and move to device
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Generate translations
        with torch.no_grad():
            translated = self.model.generate(**batch, max_length=512, num_beams=4)

        # Decode results
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

    def translate_list(
        self, text_list: List[str], desc: str = "Translating"
    ) -> List[str]:
        """
        Translate a list of texts with caching and progress bar.

        Args:
            text_list: List of texts to translate
            desc: Description for progress bar

        Returns:
            List of translated texts in original order
        """
        # Get unique texts to avoid duplicate translations
        unique_texts = list(set(text_list))
        translated_map = {}

        print(f"{desc}: {len(unique_texts)} unique texts from {len(text_list)} total")

        # Process in batches
        for i in tqdm(range(0, len(unique_texts), self.batch_size), desc=desc):
            batch = unique_texts[i : i + self.batch_size]

            try:
                translations = self.translate_batch(batch)
                translated_map.update(dict(zip(batch, translations)))

                # Clear GPU cache periodically
                if self.device.type == "cuda" and i % (self.batch_size * 10) == 0:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(
                    f"GPU OOM at batch {i}. Falling back to single-item processing..."
                )
                torch.cuda.empty_cache()

                # Process one by one if OOM
                for text in batch:
                    translation = self.translate_batch([text])
                    translated_map[text] = translation[0]

        # Return translations in original order
        return [translated_map[text] for text in text_list]

    def translate_visdial(self, visdial_data: Dict) -> Dict:
        """
        Translate a complete VisDial dataset.

        Args:
            visdial_data: VisDial dataset dictionary

        Returns:
            Translated VisDial dataset
        """
        data = visdial_data["data"]

        # Extract components
        questions = data["questions"]
        answers = data["answers"]
        dialogs = data["dialogs"]

        print(f"\nDataset statistics:")
        print(f"- Questions: {len(questions)}")
        print(f"- Answers: {len(answers)}")
        print(f"- Dialogs: {len(dialogs)}")

        # Translate questions and answers
        translated_questions = self.translate_list(questions, "Translating questions")
        translated_answers = self.translate_list(answers, "Translating answers")

        # Translate captions
        captions = [dialog["caption"] for dialog in dialogs]
        translated_captions = self.translate_list(captions, "Translating captions")

        # Update dialogs with translated captions
        for i, dialog in enumerate(dialogs):
            dialog["caption"] = translated_captions[i]

        # Update data
        data["questions"] = translated_questions
        data["answers"] = translated_answers
        data["dialogs"] = dialogs

        # Return updated dataset
        visdial_data["data"] = data
        return visdial_data


def main():
    parser = argparse.ArgumentParser(
        description="Translate VisDial datasets to Portuguese or Spanish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["pt", "es"],
        default="es",
        help="Target language: pt (Portuguese) or es (Spanish)",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=".",
        help="Input directory containing original VisDial JSON files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="translated",
        help="Output directory for translated files",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for translation (default: 32)",
    )

    parser.add_argument(
        "--splits",
        "-s",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to translate",
    )

    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Directory to cache models"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize translator
    translator = VisDialTranslator(
        target_language=args.language,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        cache_dir=args.cache_dir,
    )

    # Process each split
    lang_suffix = args.language.upper()

    for split in args.splits:
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*50}")

        input_file = Path(args.input_dir) / f"visdial_1.0_{split}.json"
        output_file = output_dir / f"visdial_1.0_{split}_{lang_suffix}.json"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue

        try:
            # Load data
            print(f"Loading: {input_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                visdial_data = json.load(f)

            # Translate
            translated_data = translator.translate_visdial(visdial_data)

            # Save
            print(f"Saving: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)

            print(f"✓ Successfully translated {split} split")

            # Clear GPU cache between files
            if translator.device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Error processing {split}: {e}")
            if translator.device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
