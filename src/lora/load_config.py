from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


model_path = Path(__file__).resolve().parents[2] / "models" / "Qwen3-8B"
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"Model config not found: {model_path / 'config.json'}")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=str(model_path),
    local_files_only=True,
    dtype="auto",
)
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, config)

print(model)
print(peft_model)
print(Path(__file__).resolve().parents[0])
