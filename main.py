import torch
import json

from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from parser import parse_file_to_txt

from huggingface_hub import login
login("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")

MODEL_NAME = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

peft_model_path = "adapters"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = PeftConfig.from_pretrained(peft_model_path)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config)
model = PeftModel.from_pretrained(model, peft_model_path)

app = FastAPI()

def inference(text: str, max_length: int = 3000, num_return_sequences: int = 1) -> str:
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output.replace(text, "")


@app.post("/extractInfo/")
async def extract_information(file: UploadFile = File(...)):
    data: bytes = await file.read()
    filename: str = file.filename
    text = parse_file_to_txt(data, filename)
    
    if text is None:
        return {"error": "Unsupported or empty document."}

    output = inference(text)

    try:
        output = json.loads(output)
    except Exception as e:
        print("JSON parsing error.")
        
        return {"error": str(e), "rawContent": output}

    return output
    









