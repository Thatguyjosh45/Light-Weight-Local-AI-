from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a smaller, CPU-friendly model
model_name = "distilgpt2"  # Much lighter than TinyLlama

# Optimize for CPU
torch.set_num_threads(2)  # Prevent system freeze by limiting CPU usage

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Conversation history (keep very short)
conversation_history = []

def generate_response(prompt):
    """Generates a response with minimal memory usage."""
    conversation_history.append(f"User: {prompt}")
    context = "\n".join(conversation_history[-3:])  # Only last 3 messages to save memory

    inputs = tokenizer(context, return_tensors="pt", truncation=True).to("cpu")

    # Generate response with lower max length
    outputs = model.generate(
        **inputs,
        max_length=50,  # Shorter response to save memory
        do_sample=False  # More predictable output, avoids excessive computation
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    conversation_history.append(f"AI: {response}")

    return response

# Interactive chat loop
print("Lightweight Chatbot! Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = generate_response(user_input)
    print("AI:", response)
