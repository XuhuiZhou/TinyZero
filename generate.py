from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    # Path to your fine-tuned model
    model_path = "/usr2/xuhuiz/models/Qwen2.5-0.5B/checkpoints/TinyZero/countdown-qwen2.5-0.5b/actor/global_step_200"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Automatically choose best device setup
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use fp16 for efficiency
    )
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=256):
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Example prompt
    prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers [79, 17, 60], create an equation that equals 36. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"
    
    # Generate text
    generated_text = generate_text(prompt, model, tokenizer)
    print("Generated text:")
    print(generated_text) 