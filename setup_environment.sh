#!/bin/bash
echo "Setting up Python 3.11 Virtual Environment..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 not found! Please install it first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install packages
echo "Installing required packages..."
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

echo
echo "Virtual environment setup complete!"
echo "You can now run: python employee_analysis.py"