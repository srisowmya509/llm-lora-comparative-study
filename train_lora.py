from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

# TinyLlama model name
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


print("Loading tokenizer...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print("Loading model...")

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


print("✅ TinyLlama loaded successfully!")


# Configure LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Attach LoRA adapters
model = get_peft_model(model, lora_config)


print("\n✅ LoRA adapters added successfully!\n")


# Show trainable parameters
model.print_trainable_parameters()


# Load training dataset
with open("data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)


print(f"\nLoaded {len(data)} training examples.")


# Convert JSON data to Hugging Face Dataset
dataset = Dataset.from_list(data)


print("✅ Hugging Face Dataset created successfully!")


print("\n🎉 Current LoRA pipeline setup completed!")
# Tokenization function
def tokenize_function(example):
    text = (
        "Instruction: " + example["instruction"] +
        "\nResponse: " + example["response"]
    )

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function)


print("✅ Dataset tokenized successfully!")

print(tokenized_dataset)
from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./lora-tinyllama-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    fp16=False,
    use_cpu=True
)


print("✅ Training arguments created!")
# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("✅ Trainer created successfully!")
# Start training
print("\n🚀 Starting LoRA fine-tuning...\n")

trainer.train()

print("\n✅ LoRA fine-tuning completed!")