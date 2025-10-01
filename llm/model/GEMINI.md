# Gemini Workspace Analysis

## Project Overview

This repository is a collection of Python-based AI/Machine Learning projects. It includes projects for fine-tuning embedding models, and a simple linear regression model for predicting package size. The projects are well-documented and include clear instructions for setup and execution.

## Directory Breakdown

*   `embedding/`: Contains projects related to training and evaluating embedding models.
    *   `embed_trainer/`: A general-purpose fine-tuner for various Hugging Face embedding models. This is the recommended tool for fine-tuning embeddings.
    *   `qwen/`: A fine-tuner and tools optimized for the Qwen3 Embedding family.
*   `package_size_predict/`: A PyTorch project to predict package size over time using a linear regression model.

## Key Technologies

*   **Programming Languages**: Python
*   **AI/Machine Learning**:
    *   **Frameworks**: `PyTorch`, `Hugging Face Transformers`, `sentence-transformers`
    *   **Libraries**: `pandas`, `openpyxl`

## Building and Running

### Python Projects

1.  **Virtual Environment**: It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies**: Each project has its own `requirements.txt` file.
    *   **embed_trainer**: `pip install -r embedding/embed_trainer/requirements.txt`
    *   **qwen**: `bash embedding/qwen/tools/setup_venvs.sh`
    *   **package_size_predict**: `pip install -r package_size_predict/requirements.txt`

3.  **Running Scripts**:
    *   **embed_trainer**: `python embedding/embed_trainer/train.py --config embedding/embed_trainer/config.json`
    *   **package_size_predict**: `python run_training.py`

## Development Conventions

*   **Python**:
    *   Use of virtual environments is standard practice.
    *   Projects are well-documented with `README.md` files.
    *   Code is generally well-commented.
*   **General**:
    *   The project is organized into distinct modules, promoting separation of concerns.
