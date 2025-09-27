# Gemini Workspace Analysis

## Project Overview

This repository is a personal AI and game development playground. It encompasses a wide range of projects, from fundamental machine learning studies to a complete game engine and server, as well as practical AI applications like a game bot. The primary focus is on experimenting with and integrating AI technologies into various domains, particularly gaming.

## Directory Breakdown

*   **`/study`**: Contains Python scripts for learning and implementing various machine learning and deep learning concepts. This includes classification, regression, and neural network examples using libraries like `scikit-learn` and `tensorflow`.
*   **`/game-engine`**: Houses the `KojeomEngine`, a lightweight C++ game engine built on DirectX 11. It's designed with a modular and extensible architecture.
*   **`/game-server`**: Contains the `KojeomGameServer`, a C# server for the game, developed with Visual Studio.
*   **`/bot`**: Includes a Python-based bot for the mobile game "Mabinogi Mobile." It uses `ultralytics` for object detection and `pywin32` for input automation.
*   **`/llm`**: Dedicated to experiments with Large Language Models (LLMs), including agents, retrieval-augmented generation (RAG), and model tuning.
*   **`/reinforcement_learning`**: Contains projects and studies related to reinforcement learning, with connections to the Unity ML-Agents toolkit.
*   **`/documents`**: A collection of useful resources, including links to influential AI research papers.
*   **`/workflow`**: Likely contains CI/CD and automation workflows, although the specific implementation details are not immediately apparent.

## Key Technologies

*   **Programming Languages**: Python, C++, C#
*   **AI/Machine Learning**:
    *   **Frameworks**: `tensorflow`, `pytorch`, `scikit-learn`
    *   **LLM**: `llama-index`, `langchain`, `ollama`
    *   **Object Detection**: `ultralytics` (YOLO)
*   **Game Development**:
    *   **Engine**: Custom C++ engine with DirectX 11
    *   **Server**: C# with .NET
*   **Databases**: `chromadb` (likely for vector storage)
*   **DevOps**: `docker`

## Building and Running

### Python Projects (ML, Bot, LLM)

1.  **Virtual Environment**: It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv my_ai_venv
    source my_ai_venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Running Scripts**:
    *   **ML Studies**: `python study/classification_heart_data.py`
    *   **Mabinogi Bot**: `python bot/mabinogi-mobile/main_improved.py`

### C++ Game Engine (`KojeomEngine`)

*   **Requirements**:
    *   Visual Studio 2019 or newer
    *   DirectX 11 SDK
*   **Building**: Open the solution file in the `game-engine/KojeomEngine` directory and build the project.

### C# Game Server (`KojeomGameServer`)

*   **Requirements**:
    *   Visual Studio with .NET development workload
*   **Building**: Open `game-server/KojeomGameServer.sln` in Visual Studio and build the solution.

## Development Conventions

*   **Python**:
    *   Use of virtual environments is standard practice.
    *   Code is generally well-commented, with a mix of English and Korean.
*   **C++ (`KojeomEngine`)**:
    *   Follows a clear coding style with `PascalCase` for classes and functions and `camelCase` for variables.
    *   Emphasizes modern C++ practices, such as using `ComPtr` for resource management.
*   **General**:
    *   The project is organized into distinct modules, promoting separation of concerns.
    *   `README.md` files are present in several key directories, providing specific instructions and documentation.
