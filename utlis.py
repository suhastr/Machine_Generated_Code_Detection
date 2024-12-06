
import quanto.quantize
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    QuantoConfig
)
import quanto
import torch
from typing import Tuple

def load_model_and_tokenizer(model_path: str, auth_token: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer from the specified model path.

    Args:
        model_path (str): Path to the pre-trained model.
        auth_token (str, optional): Hugging Face authentication token. Defaults to None.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer.
    """
    print(f"Initializing tokenizer from: {model_path}")
    
    # Load the tokenizer from the pre-trained model path
    tokenizer_instance = AutoTokenizer.from_pretrained(model_path, token=auth_token)
    if tokenizer_instance.pad_token is None:
        tokenizer_instance.pad_token = tokenizer_instance.eos_token
        print("Pad token was not set. Defaulting to EOS token: ", tokenizer_instance.pad_token)
    print("Tokenizer loaded successfully.")
    
    # Check for CUDA availability and inform the user
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU for computation.")
    else:
        print("CUDA not available. Defaulting to CPU.")
    
    # Load the pre-trained base model
    print("Loading base model...")
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=auth_token,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Base model loaded successfully.")
    
    return model_instance, tokenizer_instance
