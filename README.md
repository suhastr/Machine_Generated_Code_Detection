# Machine Generated Code Detection

This repository provides a solution for detecting machine-generated code using AI-based models. It employs pretrained language models and fine-tuning techniques to analyze whether a given piece of code is AI-generated or human-written.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Folder and File Structure](#folder-and-file-structure)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)

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

