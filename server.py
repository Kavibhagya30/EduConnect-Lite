from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load Gemma-2B Model and Tokenizer
model_path = "D:/models--google--gemma-2b/snapshots/9cf48e52b224239de00d483ec8eb84fb8d0f3a3a"  # Update with your actual path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# ✅ Move Model to GPU if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Enable CORS for Web Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],
)

# ✅ Define Request Model
class RequestData(BaseModel):
    prompt: str

# ✅ API Endpoint to Generate Responses
@app.post("/generate/")
async def generate_text(request: RequestData):
    try:
        input_text = request.prompt
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Generate response
        output_ids = model.generate(input_ids, max_length=256, temperature=0.7, top_p=0.9, do_sample=True)
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Root Endpoint (For Testing)
@app.get("/")
async def root():
    return {"message": "Gemma-2B FastAPI server is running!"}
