#!/usr/bin/env python
import torch
import transformers

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

_tokenizer = None
_pipe = None

def load_model():
    """Loads the model and tokenizer if not already loaded."""
    global _tokenizer, _pipe
    if _tokenizer is None or _pipe is None:
        _tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=_tokenizer,
        )
    return _tokenizer, _pipe

def summarize_text(text, max_tokens=1200):
    """
    Summarizes the provided text using Llama 3.1 Instruct.
    """
    tokenizer, pipe = load_model()
    system_message = (
        "You are a helpful assistant that summarizes text comprehensively. "
        "Include examples from the text when necessary."
    )
    user_message = (
        f"{text}\n\n"
        "Please provide a comprehensive summary of the above text, "
        "including any examples that help illustrate its points."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=0.5,
        do_sample=True,
        num_return_sequences=1,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0]["generated_text"]
    return generated
