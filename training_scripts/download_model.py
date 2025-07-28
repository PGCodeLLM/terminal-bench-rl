from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil

model_id = os.getenv('HF_MODEL_NAME', 'Qwen/Qwen3-32B')
# Replace slashes with underscores to avoid nested directories
model_name = model_id.replace('/', '_')
model_dir = Path(f'./models/{model_name}')

# Check if the model is already properly saved
if model_dir.exists() and (model_dir / 'config.json').exists():
    print(f'{model_id} already exists at {model_dir}')
    sys.exit(0)

# Create directory if it doesn't exist
model_dir.parent.mkdir(parents=True, exist_ok=True)

print(f'Downloading model {model_id}...')

# Download and save the model properly
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Save to the correct location
print(f'Saving model to {model_dir}...')
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f'Model saved to {model_dir}')
