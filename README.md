# GPT-2 Fine-Tuning and Text Generation

This repository provides a simple implementation for fine-tuning OpenAI's GPT-2 model on the `openwebtext` dataset and using it for text generation. The implementation is done using PyTorch and Hugging Face's `transformers` library.

## Features
- Load and fine-tune a pre-trained GPT-2 model.
- Generate text based on a given prompt.
- Chat interactively with the fine-tuned model.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install torch transformers datasets
```

## Usage

### 1. Load Pre-trained GPT-2 Model
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 2. Generate Text
```python
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time,"
generated_text = generate_text(prompt)
print(generated_text)
```

### 3. Load Dataset
```python
from datasets import load_dataset

dataset = load_dataset("openwebtext", split="train")
data = [sample["text"] for sample in dataset.select(range(100))]  # Selecting a small subset
```

### 4. Fine-Tune GPT-2
```python
import torch.optim as optim
import torch.nn as nn

def fine_tune(model, data, epochs=1, lr=5e-5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for text in data:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    return model

# Start fine-tuning
fine_tune(model, data)
```

### 5. Generate Response for Chatbot
```python
def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    output = model.generate(input_ids, attention_mask=attention_mask,
                            max_length=max_length, num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=0.7, top_k=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### 6. Interactive Chatbot
```python
def chat_with_bot():
    print("Hello! I'm your chatbot. Type 'quit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Run chat
chat_with_bot()
```

## Notes
- The dataset is loaded using the `datasets` library, which provides easy access to large-scale datasets.
- The fine-tuning function uses an AdamW optimizer and cross-entropy loss for training.
- The chatbot implementation allows interactive conversations with the model.

## License
This project is open-source and available under the MIT License.

