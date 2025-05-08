from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import vllm
import os

app = FastAPI()

# Initialize vLLM model
model_id = os.environ.get("MODEL_ID", "codellama/CodeLlama-7b-Instruct-hf")
model = vllm.LLM(model_id)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

@app.post("/generate")
async def generate(request: GenerationRequest):
    sampling_params = vllm.SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    outputs = model.generate([request.prompt], sampling_params)
    return {"results": [{"text": output.outputs[0].text} for output in outputs]}

@app.get("/")
async def root():
    return {"message": "vLLM API is running! Use /generate endpoint for text generation."}
