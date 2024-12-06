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
- [Challenges](#challenges)
- [Hardware Resource](#hardware-resource)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [References](#references)

---

## Overview

The project leverages transformer models from Hugging Face to determine the origin of code (machine-generated vs. human-written). It provides a Flask-based backend for serving the analysis and a minimalistic HTML frontend for interacting with the API. This project builds upon the research presented in the paper [Binoculars](https://arxiv.org/pdf/2401.12070). While the original implementation of Binoculars lacked the capability to detect AI-generated code, we have extended its functionality to include robust AI code detection.

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
3. GPU support must for running large models.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/Machine_Generated_Code_Detection.git
    cd Machine_Generated_Code_Detection
    ```
    
3. Create .gitignore file:
    ```bash
    # Ignore Python virtual environments
    venv/
    __pycache__/

    # Ignore Hugging Face token
    hugging_face_auth_token.txt
    ```
    
3. Create a virtual environment:
    ```bash
    python3 -m venv env_name
    ```
    
4. Activate the Virtual Environment:
    ```bash
    source env_name/bin/activate
    ```

5. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

6. Add your Hugging Face authentication token:
    - Save the token in the `hugging_face_auth_token.txt` file.

---

## Usage

### Fine tuning the model
1. Open the file model_finetuning.py and make below changes [optional]
1. Select Model and Dataset of your choice
    ```bash
    # MODEL_TO_FINETUNE = "HuggingFaceTB/SmolLM-360M"
    # MODEL_TO_FINETUNE = "HuggingFaceTB/SmolLM-360M-Instruct"
    # SAVE_NAME = "SmolLM-360M-LORA"

    # FINETUNE_DATASET = "ise-uiuc/Magicoder-Evol-Instruct-110K"
    # FINETUNE_DATASET = "bigcode/starcoderdata"
    # FINETUNE_DATASET = "iamtarun/code_instructions_120k_alpaca"
    ```
3. Set the number of epoch of your choice
4. Execute the file
   ```bash
    python model_tuning.py
    ```
5. After the model finishes fine tuning it is saves the model under fine_tuned_model, creates results and log directories with content.
   ![fine_tuned](https://github.com/user-attachments/assets/7d4bfbfd-7eea-4b87-bc13-c18dab56abf5)

   

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
        "codeclassifier": {
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
- **Sources**: [Open-source repositories](https://github.com/IBM/Project_CodeNet?tab=readme-ov-file), GPT-generated code snippets, and curated datasets.
- **Format**: JSON or text files, where each entry contains:
  - Code snippet.
  - Label specifying if it's machine-generated (1) or human-written (0).

---

## Test

- **Integration Testing**: This test is performed by calling codeclassifier file which reads in the input from the `test_prompt.txt`.
- **System Testing**: The tests for the code detection pipeline (code_detector_validation_pipeline.py) are provided in `validate_datasets/datasetforyourchoice.csv`.
- **Test Cases**:
  - Valid machine-generated code is labelled has 1.
  - Valid human-written code is labelled has 0.


---

## Results

- The model achieves **87% accuracy** in distinguishing machine-generated code from human-written code.
![roc_auc](https://github.com/user-attachments/assets/3f24297c-f92d-4e04-8bc5-9fe6a708a512)


---

## Challenges

- **Optimizing Fine-Tuning with Limited Resources:**  
  The model fine-tuning process was constrained by limited GPU, CPU, and computational resources. As a result, we were able to fine-tune the model over a limited number of epochs.

- **Long Training Times vs. Resource Availability:**  
  Fine-tuning the model for 3 epochs required approximately 18 hours. However, the project was executed on a Hopper system, where the maximum session availability was restricted to 12 hours, presenting a significant challenge.

- **Hyperparameter Optimization and Threshold Tuning:**  
  Since the algorithms were implemented from scratch with custom improvements, determining the optimal thresholds and hyperparameters to accurately detect AI-generated content was a challenging, highly experimental process.

- **Curating High-Quality Datasets:**  
  Identifying and sourcing high-quality datasets with a balanced mix of human-generated and machine-generated code required significant effort.

- **Addressing Dataset Bias:**  
  Special attention was given to mitigating potential biases present in machine-generated code datasets to ensure fairness and accuracy in the model’s predictions.


---

## Hardware Resource

- NVIDIA A100 (80GB VRAM).
- only 1 GPU 40GB was available per session.

---

## Conclusion

This project demonstrates the feasibility of detecting machine-generated code using state-of-the-art transformer models. Future work involves refining models, expanding datasets, and deploying the solution in production environments.

---

## Contributors

- **Suhas** 
- **Manish**
- **Kashish**

---

## References

- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)

## Citation

```bibtex
@article{hans2024spotting,
  title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
  author={Hans, Abhimanyu and Schwarzschild, Avi and Cherepanova, Valeriia and Kazemi, Hamid and Saha, Aniruddha and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
  journal={arXiv preprint arXiv:2401.12070},
  year={2024}
}


