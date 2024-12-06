
import quanto.quantize
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantoConfig
import quanto
import torch
from typing import Tuple
import re

# Model configuration constants
# below is the example, it's not finetuned, you have to replace this once you fine tune the model with appropriate dataset.
# PERFORMER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
# OBSERVER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M"

# Quantization configuration for models
QUANT_CONFIG = QuantoConfig(weights="int4")


class CodeClassifier:
    def __init__(self, observer_model_name: str, performer_model_name: str, hf_api_token: str):
        """
        Initialize the CodeClassifier class with observer and performer models.
        Apply quantization for optimization using QuantoConfig.
        """
        # Load observer and performer models and their tokenizers
        self.observer_model, self.observer_tokenizer = self.load_model_and_tokenizer(observer_model_name, hf_api_token)
        self.performer_model, self.performer_tokenizer = self.load_model_and_tokenizer(performer_model_name, hf_api_token)

        # Quantize models to optimize performance
        quanto.quantize(self.performer_model, QUANT_CONFIG)
        quanto.quantize(self.observer_model, QUANT_CONFIG)

    def classify_code(self, input_text: str, device, threshold_low=20, threshold_medium=25, threshold_high=34) -> Tuple[str, float]:
        """
        classify_code whether a given text is AI-generated based on its score.
        """
        score = self.calculate_score(input_text, device)
        evaluation_result = ""

        # Determine the AI generation status based on thresholds
        if score >= threshold_high:
            evaluation_result = "Definitely AI-generated"
        elif threshold_high > score >= threshold_medium:
            evaluation_result = "Probably AI-generated"
        elif threshold_medium > score >= threshold_low:
            evaluation_result = "Probably not AI-generated"
        else:
            evaluation_result = "Not AI-generated"

        return evaluation_result, score

    def calculate_score(self, input_text: str, device) -> float:
        """
        Compute a score for the input text using perplexity calculations.
        """
        perplexity_score = self.calculate_perplexity(
            input_text, 
            self.performer_model, 
            self.observer_model, 
            self.performer_tokenizer, 
            device
        )
        return perplexity_score.cpu()[0]

    @torch.no_grad()
    def calculate_log_perplexity(self, input_text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device):
        """
        Compute the log perplexity of the input text using the specified model and tokenizer.
        """
        text_tokens = tokenizer(input_text, return_tensors="pt").to(device)
        total_log_likelihood = 0
        token_count = 0

        # Initialize empty context tokens
        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)

        for token in text_tokens['input_ids'][0]:
            # Update context and compute the next token logits
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
            model_outputs = model(context_tokens)
            next_token_logits = model_outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            total_log_likelihood -= torch.log(next_token_probs[:, token])
            token_count += 1

        return total_log_likelihood / token_count

    @torch.no_grad()
    def calculate_cross_perplexity(
        self, input_text: str, performer_model: AutoModelForCausalLM, observer_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, device
    ):
        """
        Compute cross perplexity using both performer and observer models.
        """
        text_tokens = tokenizer(input_text, return_tensors="pt").to(device)
        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)

        total_cross_entropy = 0
        token_count = 0

        for token in text_tokens['input_ids'][0]:
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)

            # Compute logits from both models
            performer_outputs = performer_model(context_tokens)
            observer_outputs = observer_model(context_tokens)

            performer_logits = performer_outputs.logits[:, -1, :]
            observer_logits = observer_outputs.logits[:, -1, :]

            # Compute cross entropy
            performer_probs = torch.softmax(performer_logits, dim=-1)
            observer_probs = torch.softmax(observer_logits, dim=-1)

            total_cross_entropy -= torch.matmul(performer_probs, torch.log(observer_probs).T)

        return total_cross_entropy / token_count

    @torch.no_grad()
    def calculate_perplexity(
        self, input_text: str, performer_model: AutoModelForCausalLM, observer_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, device
    ):
        """
        Compute perplexity using a combination of normal and cross-perplexity methods.
        """
        text_tokens = tokenizer(input_text, return_tensors="pt").to(device)
        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)

        cross_entropy_cp = 0
        cross_entropy_np = 0

        for token in text_tokens['input_ids'][0]:
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)

            # Model outputs
            performer_outputs = performer_model(context_tokens)
            observer_outputs = observer_model(context_tokens)

            performer_logits = performer_outputs.logits[:, -1, :]
            observer_logits = observer_outputs.logits[:, -1, :]

            # Adjust logits using temperature
            temperature = 2.0
            performer_logits = (performer_logits / temperature).to(torch.float32)
            observer_logits = (observer_logits / temperature).to(torch.float32)
            performer_logits -= performer_logits.max(dim=-1, keepdim=True).values
            observer_logits -= observer_logits.max(dim=-1, keepdim=True).values

            # Compute softmax
            performer_probs = torch.softmax(performer_logits, dim=-1)
            observer_probs = torch.softmax(observer_logits, dim=-1)

            # Calculate entropy terms
            cross_entropy_cp -= torch.matmul(performer_probs, torch.log(observer_probs).T)
            cross_entropy_np -= torch.log(performer_probs[:, token])

        if cross_entropy_cp == 0:
            print("Error: Division by zero in perplexity calculation.")
            return torch.tensor(float('nan'))

        return cross_entropy_np / cross_entropy_cp

    def load_model_and_tokenizer(self, model_path: str, hf_api_token: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and its tokenizer from a specified path.
        """
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_api_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Pad token set to EOS token: ", tokenizer.pad_token)
        print("Tokenizer loaded successfully")

        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, token=hf_api_token, torch_dtype=torch.float16, device_map="auto"
        )
        print("Base model loaded successfully")
        return model, tokenizer


def load_hf_auth_token(auth_token_file: str):
    """
    Retrieve the Hugging Face authentication token from a file.
    """
    with open(auth_token_file) as file:
        return file.readline().strip()


if __name__ == "__main__":
    # Initialize the CodeClassifier class and test the classify_codeion method
    hf_auth_token = load_hf_auth_token("hugging_face_auth_token.txt")
    CodeClassifier_instance = CodeClassifier(OBSERVER_MODEL_NAME, PERFORMER_MODEL_NAME, hf_auth_token)

    with open("test_prompt.txt") as file:
        test_text = "\n".join(file.readlines())

    ai_generated_status, confidence = CodeClassifier_instance.classify_code(test_text, "cuda:0")
    print(f"AI-Generated Status: {ai_generated_status}")
    print(f"Confidence Score: {confidence}")
