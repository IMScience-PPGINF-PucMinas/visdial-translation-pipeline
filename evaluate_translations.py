#!/usr/bin/env python3
"""
VisDial Translation Evaluation Script
Manual evaluation tool for assessing translation quality of VisDial datasets
Based on the paper: "Multilingual Visual Understanding" (CIARP 2025)
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class TranslationEvaluator:
    """Manual evaluation tool for VisDial translations."""

    LANGUAGES = {
        "pt": "Portuguese",
        "es": "Spanish",
    }

    def __init__(self, language: str, input_dir: str = "."):
        """
        Initialize the evaluator.

        Args:
            language: Target language code ('pt' or 'es')
            input_dir: Directory containing VisDial JSON files
        """
        if language not in self.LANGUAGES:
            raise ValueError(
                f"Language {language} not supported. Choose from: {list(self.LANGUAGES.keys())}"
            )

        self.language = language
        self.language_name = self.LANGUAGES[language]
        self.input_dir = Path(input_dir)

    def load_datasets(self, split: str = "val") -> Tuple[Dict, Dict]:
        """
        Load original and translated datasets.

        Args:
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (original_data, translated_data)
        """
        # Load original English dataset
        english_file = self.input_dir / f"visdial_1.0_{split}.json"
        if not english_file.exists():
            raise FileNotFoundError(f"English file not found: {english_file}")

        print(f"Loading original English dataset: {english_file}")
        with open(english_file, "r", encoding="utf-8") as f:
            english_data = json.load(f)

        # Load translated dataset
        lang_suffix = self.language.upper()
        translated_file = self.input_dir / f"visdial_1.0_{split}_{lang_suffix}.json"
        if not translated_file.exists():
            raise FileNotFoundError(f"Translated file not found: {translated_file}")

        print(f"Loading translated {self.language_name} dataset: {translated_file}")
        with open(translated_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)

        return english_data, translated_data

    def sample_examples(
        self, english_data: Dict, translated_data: Dict, n_samples: int
    ) -> List[Dict]:
        """
        Randomly sample examples for evaluation.

        Args:
            english_data: Original English dataset
            translated_data: Translated dataset
            n_samples: Number of samples to extract

        Returns:
            List of sample dictionaries containing original and translated text
        """
        samples = []

        # Get data sections
        en_data = english_data["data"]
        tr_data = translated_data["data"]

        en_questions = en_data["questions"]
        tr_questions = tr_data["questions"]

        en_answers = en_data["answers"]
        tr_answers = tr_data["answers"]

        en_dialogs = en_data["dialogs"]
        tr_dialogs = tr_data["dialogs"]

        # Sample questions
        question_indices = random.sample(
            range(len(en_questions)), min(n_samples // 3, len(en_questions))
        )
        for idx in question_indices:
            samples.append(
                {
                    "type": "Question",
                    "english": en_questions[idx],
                    "translated": tr_questions[idx],
                }
            )

        # Sample answers
        answer_indices = random.sample(
            range(len(en_answers)), min(n_samples // 3, len(en_answers))
        )
        for idx in answer_indices:
            samples.append(
                {
                    "type": "Answer",
                    "english": en_answers[idx],
                    "translated": tr_answers[idx],
                }
            )

        # Sample captions
        caption_indices = random.sample(
            range(len(en_dialogs)), min(n_samples // 3, len(en_dialogs))
        )
        for idx in caption_indices:
            samples.append(
                {
                    "type": "Caption",
                    "english": en_dialogs[idx]["caption"],
                    "translated": tr_dialogs[idx]["caption"],
                }
            )

        # Shuffle samples
        random.shuffle(samples)

        # Trim to exact number if needed
        return samples[:n_samples]

    def evaluate_sample(self, sample: Dict, sample_num: int, total: int) -> Dict:
        """
        Display a sample and collect evaluation scores.

        Args:
            sample: Sample dictionary
            sample_num: Current sample number
            total: Total number of samples

        Returns:
            Dictionary with evaluation scores
        """
        print("\n" + "=" * 80)
        print(f"SAMPLE {sample_num}/{total}")
        print("=" * 80)
        print(f"\nType: {sample['type']}")
        print(f"\nEnglish (Original):")
        print(f"  {sample['english']}")
        print(f"\n{self.language_name} (Translation):")
        print(f"  {sample['translated']}")
        print("\n" + "-" * 80)

        # Evaluation criteria based on the paper
        print("\nEvaluation Criteria (1-3 scale):")
        print("  1 = Poor")
        print("  2 = Good")
        print("  3 = Excellent")
        print()

        # Get fluency score
        while True:
            try:
                fluency = int(
                    input("Fluency (grammatical correctness and naturalness) [1-3]: ")
                )
                if 1 <= fluency <= 3:
                    break
                print("Please enter a value between 1 and 3")
            except ValueError:
                print("Please enter a valid number")

        # Get adequacy score
        while True:
            try:
                adequacy = int(
                    input("Adequacy (preservation of original meaning) [1-3]: ")
                )
                if 1 <= adequacy <= 3:
                    break
                print("Please enter a value between 1 and 3")
            except ValueError:
                print("Please enter a valid number")

        # Get coherence score
        while True:
            try:
                coherence = int(
                    input("Coherence (consistency with dialog context) [1-3]: ")
                )
                if 1 <= coherence <= 3:
                    break
                print("Please enter a value between 1 and 3")
            except ValueError:
                print("Please enter a valid number")

        return {
            "type": sample["type"],
            "english": sample["english"],
            "translated": sample["translated"],
            "fluency": fluency,
            "adequacy": adequacy,
            "coherence": coherence,
        }

    def save_results(self, results: List[Dict], output_file: str):
        """
        Save evaluation results to a text file.

        Args:
            results: List of evaluation dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"VisDial Translation Evaluation Results - {self.language_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            # Calculate average scores
            if results:
                avg_fluency = sum(r["fluency"] for r in results) / len(results)
                avg_adequacy = sum(r["adequacy"] for r in results) / len(results)
                avg_coherence = sum(r["coherence"] for r in results) / len(results)

                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average Fluency:   {avg_fluency:.2f} / 3.0\n")
                f.write(f"Average Adequacy:  {avg_adequacy:.2f} / 3.0\n")
                f.write(f"Average Coherence: {avg_coherence:.2f} / 3.0\n")
                f.write("\n")

                # Distribution by type
                type_counts = {}
                type_scores = {}
                for result in results:
                    t = result["type"]
                    if t not in type_counts:
                        type_counts[t] = 0
                        type_scores[t] = {"fluency": 0, "adequacy": 0, "coherence": 0}
                    type_counts[t] += 1
                    type_scores[t]["fluency"] += result["fluency"]
                    type_scores[t]["adequacy"] += result["adequacy"]
                    type_scores[t]["coherence"] += result["coherence"]

                f.write("SCORES BY TYPE\n")
                f.write("-" * 80 + "\n")
                for t in type_counts:
                    count = type_counts[t]
                    f.write(f"\n{t} (n={count}):\n")
                    f.write(f"  Fluency:   {type_scores[t]['fluency']/count:.2f}\n")
                    f.write(f"  Adequacy:  {type_scores[t]['adequacy']/count:.2f}\n")
                    f.write(f"  Coherence: {type_scores[t]['coherence']/count:.2f}\n")

                f.write("\n")

            # Write individual results
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"SAMPLE {i}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Type: {result['type']}\n\n")
                f.write(f"English (Original):\n")
                f.write(f"  {result['english']}\n\n")
                f.write(f"{self.language_name} (Translation):\n")
                f.write(f"  {result['translated']}\n\n")
                f.write(f"Scores:\n")
                f.write(f"  Fluency:   {result['fluency']}/3\n")
                f.write(f"  Adequacy:  {result['adequacy']}/3\n")
                f.write(f"  Coherence: {result['coherence']}/3\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VisDial translation quality through manual assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["pt", "es"],
        required=True,
        help="Target language: pt (Portuguese) or es (Spanish)",
    )

    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=".",
        help="Directory containing VisDial JSON files (default: current directory)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)",
    )

    parser.add_argument(
        "--split",
        "-s",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate (default: val)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Auto-generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_{args.language}_{args.split}_{timestamp}.txt"

    print(f"\n{'='*80}")
    print(f"VisDial Translation Evaluation Tool")
    print(f"{'='*80}")
    print(f"Language: {args.language.upper()}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")

    # Initialize evaluator
    evaluator = TranslationEvaluator(args.language, args.input_dir)

    # Load datasets
    try:
        english_data, translated_data = evaluator.load_datasets(args.split)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the VisDial files are in the correct location:")
        print(f"  - {args.input_dir}/visdial_1.0_{args.split}.json (English original)")
        print(
            f"  - {args.input_dir}/visdial_1.0_{args.split}_{args.language.upper()}.json (Translation)"
        )
        return

    # Sample examples
    print(f"\nSampling {args.n_samples} examples for evaluation...")
    samples = evaluator.sample_examples(english_data, translated_data, args.n_samples)
    print(f"✓ Sampled {len(samples)} examples")

    # Evaluate samples
    results = []
    print("\n" + "=" * 80)
    print("BEGINNING EVALUATION")
    print("=" * 80)
    print("\nInstructions:")
    print("- Read both the original English text and the translation carefully")
    print("- Rate each translation on a scale of 1-3 for each criterion")
    print("- Press Ctrl+C to stop evaluation early and save current results")
    print()

    try:
        for i, sample in enumerate(samples, 1):
            result = evaluator.evaluate_sample(sample, i, len(samples))
            results.append(result)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("EVALUATION INTERRUPTED")
        print("=" * 80)
        print(f"Saving {len(results)} completed evaluations...")

    # Save results
    if results:
        evaluator.save_results(results, args.output)

        # Print summary
        avg_fluency = sum(r["fluency"] for r in results) / len(results)
        avg_adequacy = sum(r["adequacy"] for r in results) / len(results)
        avg_coherence = sum(r["coherence"] for r in results) / len(results)

        print(f"\nSummary ({len(results)} samples):")
        print(f"  Average Fluency:   {avg_fluency:.2f} / 3.0")
        print(f"  Average Adequacy:  {avg_adequacy:.2f} / 3.0")
        print(f"  Average Coherence: {avg_coherence:.2f} / 3.0")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
