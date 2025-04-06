import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ✅ Load Model & Tokenizer from Local Directory
model_path = r"C:\Users\KUMAR\Downloads\flan-t5-dolly-final-20250327T042231Z-001\flan-t5-dolly-final"  # Change this to your actual path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ✅ Move Model to GPU if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=512, temperature=0.9, top_p=0.95):
    """Generate an elaborate answer for a given prompt."""
    input_text = f"explain: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate response with beam search for detailed answers
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,  # Controls randomness
        top_p=top_p,  # Nucleus sampling for diversity
        num_return_sequences=1,
        do_sample=True,  # Enables non-deterministic sampling
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ✅ Example Usage
while True:
    user_input = input("\nAsk me anything (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break
    
    response = generate_response(user_input)
    print("\n🤖 Answer:", response)
