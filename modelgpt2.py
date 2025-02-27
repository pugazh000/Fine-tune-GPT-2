pip install datasets

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token 

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to generate text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time,"
generated_text = generate_text(prompt)
print(generated_text)

# Load dataset
dataset = load_dataset("openwebtext", split="train")
data = [sample["text"] for sample in dataset.select(range(100))]  # Select a small subset for testing

# Define a simple fine-tuning function
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

#for fine-tuning
fine_tune(model, data)

#to generate response ASAP
def generate_response(prompt, max_length=100):
    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Create attention mask (1 for real tokens, 0 for padding tokens)
    attention_mask = torch.ones(input_ids.shape, device=device)
    
    # Set the pad token id to the EOS token id

    output = model.generate(input_ids, attention_mask=attention_mask, 
                        max_length=max_length, num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id, 
                        temperature=0.7, top_k=50, do_sample=True)

    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Chat with the chatbot interactively
def chat_with_bot():
    print("Hello! I'm your chatbot. Type 'quit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Run the chat in Colab
chat_with_bot()
