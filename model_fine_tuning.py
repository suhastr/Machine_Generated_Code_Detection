from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

# Step 1: Load Dataset
print("Loading dataset...")
instruction_dataset = load_dataset('iamtarun/code_instructions_120k_alpaca')

# Step 2: Split Dataset into Training and Validation
print("Splitting dataset...")
split_dataset = instruction_dataset["train"].train_test_split(test_size=0.1)
training_data = split_dataset["train"]
validation_data = split_dataset["test"]

# Step 3: Load Pretrained Tokenizer
print("Loading tokenizer...")
instruction_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct')

# Step 4: Preprocess the Data
print("Tokenizing dataset...")
def preprocess_samples(examples):
    """
    Preprocess dataset samples by tokenizing instructions, inputs, and outputs.
    """
    # Create input text by combining instructions and inputs
    input_texts = [
        f"Instruction: {instruction} Input: {input_text}" 
        for instruction, input_text in zip(examples['instruction'], examples['input'])
    ]
    # Tokenize input text
    tokenized_inputs = instruction_tokenizer(input_texts, max_length=512, truncation=True, padding="max_length")

    # Tokenize output text
    with instruction_tokenizer.as_target_tokenizer():
        tokenized_outputs = instruction_tokenizer(
            examples['output'], max_length=512, truncation=True, padding="max_length"
        )
    # Add tokenized outputs as labels
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs

# Apply preprocessing to training and validation datasets
training_data = training_data.map(preprocess_samples, batched=True, num_proc=10)  # Use 10 cores for tokenization
validation_data = validation_data.map(preprocess_samples, batched=True, num_proc=10)  # Use 10 cores for tokenization

# Step 5: Load Pretrained Model
print("Loading model...")
instruction_model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct')

# Step 6: Define Training Arguments
print("Setting up training arguments...")
training_config = TrainingArguments(
    output_dir="./results",                  # Directory to save model checkpoints
    overwrite_output_dir=True,              # Overwrite existing checkpoints if any
    evaluation_strategy="epoch",            # Evaluate at the end of each epoch
    save_strategy="epoch",                  # Save model at the end of each epoch
    per_device_train_batch_size=2,         # Batch size for training
    per_device_eval_batch_size=2,          # Batch size for evaluation
    num_train_epochs=1,                     # Number of training epochs
    logging_dir="./logs",                   # Directory for logging
    logging_steps=100,                      # Log every 100 steps
    fp16=True,                              # Enable mixed precision training
    save_total_limit=2,                     # Keep only the last 2 checkpoints
    learning_rate=5e-5,                     # Learning rate
    weight_decay=0.01,                      # Apply weight decay for regularization
    dataloader_num_workers=10,              # Use 10 CPU threads for data loading
    report_to="none",                       # Disable reporting to external tools (e.g., WandB)
)

# Step 7: Initialize Trainer
print("Initializing trainer...")
model_trainer = Trainer(
    model=instruction_model,
    args=training_config,
    train_dataset=training_data,
    eval_dataset=validation_data,
)

# Step 8: Train the Model
print("Starting training...")
model_trainer.train()

# Step 9: Save the Fine-Tuned Model
fine_tuned_model_dir = "./fine_tuned_model"
print(f"Saving the fine-tuned model to {fine_tuned_model_dir}...")
model_trainer.save_model(fine_tuned_model_dir)
instruction_tokenizer.save_pretrained(fine_tuned_model_dir)

print("Model fine-tuning and saving completed.")
