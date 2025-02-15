from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B"  # or whichever Qwen2 model you're using
save_path = "./models/Qwen2.5-3B"

# Download and save the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)