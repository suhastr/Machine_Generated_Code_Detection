# Machine Generated Code Detection

This repository provides a solution for detecting machine-generated code using AI-based models. It employs pretrained language models and fine-tuning techniques to analyze whether a given piece of code is AI-generated or human-written.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Folder and File Structure](#folder-and-file-structure)
- [API Endpoints](#api-endpoints)
- [Datasets](#datasets)
  - [Dataset Description](#dataset-description)
- [Test](#test)
- [Results](#results)
- [Research Paper](#research-paper)
- [Challenges](#challenges)
- [Hardware Resource](#hardware-resource)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [References](#references)

---

## Overview

The project leverages transformer models from Hugging Face to determine the origin of code (machine-generated vs. human-written). It provides a Flask-based backend for serving the analysis and a minimalistic HTML frontend for interacting with the API.

---

## Features

- **Model Integration**: Uses Hugging Face's pretrained models (e.g., `SmolLM-360M`) for analysis.
- **Frontend**: A simple HTML page to upload code and display the results.
- **Backend API**: Flask server that processes the requests and returns AI analysis results.
- **Custom Model Fine-Tuning**: Scripts for fine-tuning the models using specific datasets.
- **Cross-Origin Resource Sharing (CORS)**: Enables integration with external services.

---

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher.
2. A valid Hugging Face authentication token.
3. GPU support for running large models (optional but recommended).

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/Machine_Generated_Code_Detection.git
    cd Machine_Generated_Code_Detection
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your Hugging Face authentication token:
    - Save the token in the `hugging_face_auth_token.txt` file.

---

## Usage

### Running the Server
1. Start the Flask server:
    ```bash
    python backend.py
    ```
2. Open the frontend in a browser:
    - The server runs by default on `http://localhost:5000`.

### Frontend
- Paste the code you want to analyze into the text box and click "Analyze Code".
- The result will display whether the code is AI-generated, along with the confidence score.

---

## Folder and File Structure


---

## API Endpoints

### `/analyze`
- **Method**: `POST`
- **Description**: Analyze the submitted code to determine if it is machine-generated.
- **Request Format**:
    ```json
    {
        "content": "<code to analyze>",
        "type": "code"
    }
    ```
- **Response Format**:
    ```json
    {
        "codedetector": {
            "is_ai_generated": "yes/no",
            "score": 0.95,
            "result": "AI Generated (Score: 0.9500)"
        }
    }
    ```

---

## Datasets

### Dataset Description
- The project uses datasets containing human-written and machine-generated code for model training and validation.
- **Sources**: Open-source repositories, GPT-generated code snippets, and curated datasets.
- **Format**: JSON or text files, where each entry contains:
  - Code snippet.
  - Metadata specifying if it's machine-generated or human-written.

---

## Test

- **Testing Framework**: The tests for the code detection pipeline are provided in `test_prompt.txt`.
- **Test Cases**:
  - Valid machine-generated code.
  - Valid human-written code.
  - Edge cases like empty inputs or highly obfuscated code.

---

## Results

- The model achieves **X% accuracy** in distinguishing machine-generated code from human-written code.
- Average confidence score for correct predictions: **Y%**.
- Evaluation metrics:
  - Precision: **A%**
  - Recall: **B%**
  - F1-score: **C%**

---

## Research Paper

If applicable, include a link to a research paper describing the methodology, experiments, and findings:
- **Paper Title**: "Detecting Machine-Generated Code with Transformer Models"
- **Authors**: Your Name, Collaborator Name
- **Publication**: Add a journal/conference or `arXiv` link.

---

## Challenges

- Fine-tuning large models with limited GPU resources.
- Identifying high-quality datasets with balanced human and machine-generated code.
- Mitigating bias in machine-generated code datasets.

---

## Hardware Resource

- **Training**: NVIDIA Tesla V100 (16GB VRAM).
- **Inference**: Tested on CPU and GPU environments.
  - Average inference time: **X seconds** on GPU, **Y seconds** on CPU.

---

## Conclusion

This project demonstrates the feasibility of detecting machine-generated code using state-of-the-art transformer models. Future work involves refining models, expanding datasets, and deploying the solution in production environments.

---

## Contributors

- **Your Name** - Initial development.
- **Collaborator Name** - Research and validation.

---

## References

- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Other relevant resources and citations.


