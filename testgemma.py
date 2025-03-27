import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Load Model & Tokenizer from the Correct Path
model_path = r"D:\models--google--gemma-2b\snapshots\9cf48e52b224239de00d483ec8eb84fb8d0f3a3a"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,  # Ensures better numerical stability
    device_map="auto"  # Uses GPU if available
)

# ✅ Function to Generate Responses
def generate_response(prompt, max_length=256, temperature=0.7, top_p=0.9):
    """Generate a relevant response from Gemma-2B."""
    formatted_prompt = f"<bos>{prompt}</s>"  # Some models need special tokens

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    output_ids = model.generate(
        **inputs, 
        max_length=max_length, 
        temperature=temperature, 
        top_p=top_p, 
        num_beams=3,  # Uses beam search for better quality
        do_sample=True
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ✅ Interactive Loop
while True:
    user_input = input("\nAsk me anything (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break
    
    response = generate_response(user_input)
    print("\n🤖 Answer:", response)
