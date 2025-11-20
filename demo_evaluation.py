#!/usr/bin/env python3
"""
Example/Demo Script for VisDial Translation Evaluation
Shows what the evaluation process looks like with sample data
"""

# Sample data demonstrating the evaluation process
SAMPLE_DATA = [
    {
        "type": "Question",
        "english": "Are there people in the room?",
        "portuguese": "Há pessoas na sala?",
        "spanish": "¿Hay personas en la habitación?",
    },
    {
        "type": "Answer",
        "english": "Yes, two people",
        "portuguese": "Sim, duas pessoas",
        "spanish": "Sí, dos personas",
    },
    {
        "type": "Caption",
        "english": "A group of people playing soccer on the field",
        "portuguese": "Um grupo de pessoas jogando futebol no campo",
        "spanish": "Un grupo de personas jugando fútbol en el campo",
    },
    {
        "type": "Question",
        "english": "What color is the ball?",
        "portuguese": "Qual é a cor da bola?",
        "spanish": "¿De qué color es la pelota?",
    },
    {
        "type": "Answer",
        "english": "Black with white Mickey logos",
        "portuguese": "Preto com logotipos do Mickey brancos",
        "spanish": "Negro con logos de Mickey blancos",
    },
]


def demonstrate_evaluation():
    """Demonstrate the evaluation process."""
    print("=" * 80)
    print("VisDial Translation Evaluation - DEMO")
    print("=" * 80)
    print("\nThis demo shows what the evaluation process looks like.")
    print("In actual use, you would be scoring real translations from the dataset.")
    print()

    # Show Portuguese examples
    print("\n" + "=" * 80)
    print("PORTUGUESE EXAMPLES")
    print("=" * 80)

    for i, sample in enumerate(SAMPLE_DATA, 1):
        print(f"\n--- Example {i} ---")
        print(f"Type: {sample['type']}")
        print(f"\nEnglish:    {sample['english']}")
        print(f"Portuguese: {sample['portuguese']}")
        print("\nYou would rate this on:")
        print("  - Fluency (1-3): How natural and grammatically correct?")
        print("  - Adequacy (1-3): Does it preserve the original meaning?")
        print("  - Coherence (1-3): Does it fit the dialog context?")

    # Show Spanish examples
    print("\n" + "=" * 80)
    print("SPANISH EXAMPLES")
    print("=" * 80)

    for i, sample in enumerate(SAMPLE_DATA, 1):
        print(f"\n--- Example {i} ---")
        print(f"Type: {sample['type']}")
        print(f"\nEnglish: {sample['english']}")
        print(f"Spanish: {sample['spanish']}")
        print("\nYou would rate this on:")
        print("  - Fluency (1-3): How natural and grammatically correct?")
        print("  - Adequacy (1-3): Does it preserve the original meaning?")
        print("  - Coherence (1-3): Does it fit the dialog context?")

    # Show example scores
    print("\n" + "=" * 80)
    print("EXAMPLE EVALUATION")
    print("=" * 80)

    example = SAMPLE_DATA[0]
    print(f"\nType: {example['type']}")
    print(f"\nEnglish:    {example['english']}")
    print(f"Portuguese: {example['portuguese']}")
    print("\nExample Scores:")
    print("  Fluency:   3/3 (Excellent - very natural Portuguese)")
    print("  Adequacy:  3/3 (Excellent - meaning fully preserved)")
    print("  Coherence: 3/3 (Excellent - fits context perfectly)")

    print("\n" + "=" * 80)
    print("TO RUN ACTUAL EVALUATION:")
    print("=" * 80)
    print("\npython evaluate_translations.py --language pt --n-samples 10")
    print("python evaluate_translations.py --language es --n-samples 10")
    print("\nThis will randomly sample actual translations from your dataset")
    print("and guide you through the evaluation process.")
    print()


if __name__ == "__main__":
    demonstrate_evaluation()
