from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

# Model loading with HuggingFace format (user/model path)
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the model and tokenizer
logger.info(f"Loading model from Hugging Face Hub: {model_path}")

# Check if CUDA is available and set the appropriate device
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# FastAPI endpoint for chat
@app.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json()
        prompt = data.get("message", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        logger.info(f"Processing prompt: {prompt[:30]}...")
        output = pipe(prompt, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.7)[0]["generated_text"]
        return {"response": output}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Chat API is running! Send POST requests to /chat"}