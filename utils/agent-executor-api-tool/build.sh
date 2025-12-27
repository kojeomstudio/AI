#!/bin/bash
# ============================================
# Agent Executor API - Linux/Mac Build Script
# ============================================
# This script builds a standalone executable using PyInstaller
# with automatic virtual environment management

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_separator() {
    echo "========================================================================"
}

print_line() {
    echo "------------------------------------------------------------------------"
}

# Start build process
echo ""
print_separator
echo "                  Agent Executor API - Build Script"
print_separator
echo ""
log_info "Starting build process..."
log_info "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================
# Step 1: Check Python installation
# ============================================
log_step "[1/7] Checking Python installation..."
print_line

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed or not in PATH"
    log_error "Please install Python 3.10 or higher"
    log_error "Visit: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
log_success "Found $PYTHON_VERSION"
echo ""

# ============================================
# Step 2: Check/Create virtual environment
# ============================================
log_step "[2/7] Checking virtual environment..."
print_line

if [ -d "venv" ]; then
    log_info "Virtual environment already exists at: $(pwd)/venv"
    log_info "Checking if it's valid..."

    if [ -f "venv/bin/python" ]; then
        log_success "Virtual environment is valid"
    else
        log_warn "Virtual environment is corrupted, recreating..."
        rm -rf venv
        log_info "Creating new virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created successfully"
    fi
else
    log_info "Creating new virtual environment..."
    python3 -m venv venv

    if [ $? -ne 0 ]; then
        log_error "Failed to create virtual environment"
        exit 1
    fi

    log_success "Virtual environment created successfully"
fi
echo ""

# ============================================
# Step 3: Activate virtual environment
# ============================================
log_step "[3/7] Activating virtual environment..."
print_line

if [ ! -f "venv/bin/activate" ]; then
    log_error "Virtual environment activation script not found"
    exit 1
fi

source venv/bin/activate
log_success "Virtual environment activated"
log_info "Python location: $(which python)"
echo ""

# ============================================
# Step 4: Upgrade pip
# ============================================
log_step "[4/7] Upgrading pip..."
print_line

python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    log_warn "Failed to upgrade pip, continuing anyway..."
else
    log_success "pip upgraded successfully"
fi

PIP_VERSION=$(pip --version)
log_info "$PIP_VERSION"
echo ""

# ============================================
# Step 5: Install dependencies
# ============================================
log_step "[5/7] Installing dependencies..."
print_line

log_info "Reading requirements from: requirements.txt"
log_info "This may take a few minutes..."
echo ""

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    log_error "Failed to install dependencies"
    log_error "Check the error messages above"
    exit 1
fi

echo ""
log_success "All dependencies installed successfully"
echo ""

# ============================================
# Step 6: Clean previous builds
# ============================================
log_step "[6/7] Cleaning previous builds..."
print_line

if [ -d "dist" ]; then
    log_info "Removing old dist directory..."
    rm -rf dist
    log_success "Old dist directory removed"
fi

if [ -d "build" ]; then
    log_info "Removing old build directory..."
    rm -rf build
    log_success "Old build directory removed"
fi

log_success "Build directories cleaned"
echo ""

# ============================================
# Step 7: Build executable with PyInstaller
# ============================================
log_step "[7/7] Building executable with PyInstaller..."
print_line

log_info "Using spec file: agent-executor-api.spec"
log_info "Build mode: One folder (with dependencies)"
log_info "This may take several minutes..."
echo ""

pyinstaller --clean --noconfirm agent-executor-api.spec

if [ $? -ne 0 ]; then
    echo ""
    log_error "Build failed"
    log_error "Check the error messages above"
    exit 1
fi

echo ""
log_success "Build completed successfully"
echo ""

# ============================================
# Step 8: Verify build output
# ============================================
print_separator
echo "                       Build Verification"
print_separator
echo ""

if [ -f "dist/agent-executor-api/agent-executor-api" ]; then
    log_success "Executable created successfully!"
    echo ""
    echo "Location: $(pwd)/dist/agent-executor-api/"
    echo "Executable: agent-executor-api"
    echo ""

    # Get file size
    SIZE=$(stat -f%z "dist/agent-executor-api/agent-executor-api" 2>/dev/null || stat -c%s "dist/agent-executor-api/agent-executor-api" 2>/dev/null)
    SIZE_MB=$((SIZE / 1048576))
    echo "File size: ${SIZE_MB} MB"

    echo ""
    log_info "Additional files in distribution:"
    ls -1 "dist/agent-executor-api" | grep -v "agent-executor-api$"

else
    log_error "Executable not found in expected location"
    log_error "Build may have failed"
    exit 1
fi

echo ""

# ============================================
# Step 9: Copy configuration files
# ============================================
print_separator
echo "                  Copying Configuration Files"
print_separator
echo ""

log_info "Copying .env.example to distribution..."
cp -f ".env.example" "dist/agent-executor-api/.env.example"
if [ $? -ne 0 ]; then
    log_warn "Failed to copy .env.example"
else
    log_success ".env.example copied"
fi

log_info "Creating default .env file..."
cp -f ".env.example" "dist/agent-executor-api/.env"
if [ $? -ne 0 ]; then
    log_warn "Failed to create .env"
else
    log_success ".env created"
fi

# Make executable runnable
chmod +x "dist/agent-executor-api/agent-executor-api"
log_success "Executable permissions set"

echo ""

# ============================================
# Final Summary
# ============================================
print_separator
echo "                         Build Summary"
print_separator
echo ""
log_success "Build completed successfully!"
echo ""
echo "Distribution directory: $(pwd)/dist/agent-executor-api/"
echo ""
echo "To run the application:"
echo "  1. Navigate to: dist/agent-executor-api/"
echo "  2. Edit .env file if needed"
echo "  3. Run: ./agent-executor-api"
echo ""
echo "To test the build:"
echo "  cd dist/agent-executor-api"
echo "  ./agent-executor-api"
echo ""
print_separator
echo ""

# Deactivate virtual environment
deactivate

exit 0
