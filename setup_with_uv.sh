#!/bin/bash
# Quick setup script using UV (fast Python package installer)

set -e  # Exit on error

# Ensure we're in the project root
cd "$(dirname "$0")"

echo "=========================================="
echo "Mamba2 Training Setup with UV"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    echo "Please run this script from /data/BioMamba/"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "${YELLOW}UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "${GREEN}✓ UV installed${NC}"
else
    echo "${GREEN}✓ UV already installed ($(uv --version))${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment with UV..."
    uv venv --python 3.10
    echo "${GREEN}✓ Virtual environment created${NC}"
else
    echo "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
echo "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "Installing dependencies with UV (this will be fast!)..."
echo ""

# Install core dependencies
uv pip install -r requirements.txt

echo ""
echo "${GREEN}✓ All dependencies installed!${NC}"

# Optional: Install recommended packages
read -p "Install recommended packages (accelerate, wandb)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing recommended packages..."
    uv pip install accelerate wandb
    echo "${GREEN}✓ Recommended packages installed${NC}"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python tests/test_setup.py

echo ""
echo "=========================================="
echo "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To start training:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run training: ./run_training.sh"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ./runs"
echo ""
