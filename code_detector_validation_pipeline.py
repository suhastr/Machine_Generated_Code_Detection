from codeclassifier import CodeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

# Model configuration constants

# below is the example, it's not finetuned, you have to replace this once you fine tune the model with appropriate dataset.


#MODEL_PERFORMER_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
#MODEL_OBSERVER_NAME = "HuggingFaceTB/SmolLM-360M"

# Load Hugging Face authentication token
with open("hugging_face_auth_token.txt") as token_file:
    AUTH_TOKEN = token_file.readline().strip()

def quantize_to_8bit(data):
    """
    Quantize the input data to 8-bit representation.
    :param data: NumPy array of input data.
    :return: Tuple of quantized data, minimum value, and scale factor.
    """
    min_value = np.min(data)
    max_value = np.max(data)
    scale_factor = (max_value - min_value) / 255
    quantized_data = np.round((data - min_value) / scale_factor).astype(np.uint8)
    return quantized_data, min_value, scale_factor

def dequantize_from_8bit(quantized_data, min_value, scale_factor):
    """
    Dequantize the 8-bit data back to its original scale.
    :param quantized_data: 8-bit quantized data.
    :param min_value: Minimum value of the original data.
    :param scale_factor: Scale factor used during quantization.
    :return: Dequantized data in original scale.
    """
    return (quantized_data.astype(np.float32) * scale_factor) + min_value

def generate_roc_curve(true_labels, scores):
    """
    Generate and save the Receiver Operating Characteristic (ROC) curve.
    :param true_labels: Ground truth binary labels.
    :param scores: Model classify_codeion scores.
    :return: Area Under the Curve (AUC) value.
    """
    # Print statistics of the data
    print(f"Number of samples: {len(true_labels)}")
    print(f"Number of positive samples: {np.sum(true_labels)}")
    print(f"Number of negative samples: {len(true_labels) - np.sum(true_labels)}")
    print(f"Min score: {np.min(scores)}")
    print(f"Max score: {np.max(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    print(f"Std score: {np.std(scores)}")

    # Handle NaN or infinity values in scores
    if np.isnan(scores).any() or np.isinf(scores).any():
        print("Warning: Scores contain NaN or infinity values. Replacing with zeros.")
        scores = np.nan_to_num(scores)

    # Calculate ROC curve and AUC
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, scores)
    roc_auc_value = auc(false_positive_rate, true_positive_rate)
    print(f'ROC AUC: {roc_auc_value}')

    # Plot ROC curve
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    return roc_auc_value

def main():
    """
    Main function to execute the analysis pipeline.
    """
    # Load the dataset and shuffle
    dataset = pd.read_csv("validate_datasets/TestDataset.csv").sample(frac=1).reset_index(drop=True)

    # Extract code and labels
    code_samples = dataset["code"]                     # column names depends on the dataset
    ai_generated_labels = dataset["generated"]

    # Initialize the CodeClassifier model
    model = CodeClassifier(MODEL_OBSERVER_NAME, MODEL_PERFORMER_NAME, AUTH_TOKEN)

    # Prepare lists to store results
    true_labels = []
    classify_codeion_scores = []

    # Iterate through the dataset and compute scores
    for code_snippet, ai_label in tqdm(zip(code_samples, ai_generated_labels), total=len(code_samples)):
        model_score = model.calculate_score(code_snippet, "cuda:0")
        print(f"AI Generated: {ai_label}, Model Score: {model_score}")
        true_labels.append(ai_label)
        classify_codeion_scores.append(model_score)

    # Convert results to NumPy arrays
    true_labels = np.array(true_labels)
    classify_codeion_scores = np.array(classify_codeion_scores)

    # Perform 8-bit quantization
    quantized_scores, minimum_value, scale = quantize_to_8bit(classify_codeion_scores)

    # Dequantize scores for evaluation and plotting
    restored_scores = dequantize_from_8bit(quantized_scores, minimum_value, scale)

    # Generate and save ROC curve
    generate_roc_curve(true_labels, restored_scores)

if __name__ == "__main__":
    main()
