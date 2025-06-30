import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_phi_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

def generate_response(tokenizer, model, query, context, chat_history):
    history_text = ""
    for turn in chat_history:
        history_text += f"<start_of_turn>user\n{turn['user']}\n<end_of_turn>\n"
        history_text += f"<start_of_turn>model\n{turn['assistant']}\n<end_of_turn>\n"

    prompt = f"""{history_text}
<start_of_turn>user
{query}

Context:
{context}
<end_of_turn>
<start_of_turn>model"""

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True)