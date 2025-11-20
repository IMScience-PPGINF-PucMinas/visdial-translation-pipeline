#!/bin/bash
# Quick start script for VisDial translation pipeline

echo "========================================"
echo "VisDial Translation Pipeline Quick Start"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Running demo..."
python3 demo_evaluation.py

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. To translate datasets:"
echo "   python3 translate_visdial.py --language pt --splits val"
echo ""
echo "2. To evaluate translations:"
echo "   python3 evaluate_translations.py --language pt --n-samples 10"
echo ""
echo "3. For more help:"
echo "   python3 translate_visdial.py --help"
echo "   python3 evaluate_translations.py --help"
echo ""
echo "See README.md for detailed documentation."
echo ""