import os
import openai
import backoff
from functools import lru_cache

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@lru_cache()
def get_llama_pipeline():
    """Load a local LLaMA model for generation.

    The model path should be provided via the ``LLAMA_MODEL_NAME`` environment
    variable. ``transformers`` must be installed. The pipeline is cached so the
    model is only loaded once.
    """
    model_name = os.getenv("LLAMA_MODEL_NAME")
    if not model_name:
        raise ValueError("LLAMA_MODEL_NAME is not set")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as e:
        raise ImportError(
            "transformers library is required for LLaMA backend") from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """Unified interface for generating text from different backends."""
    if model.startswith("llama"):
        pipe = get_llama_pipeline()
        outputs = []
        for _ in range(n):
            res = pipe(
                prompt,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )[0]["generated_text"]
            text = res[len(prompt) :]
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
                        break
            outputs.append(text.strip())
        return outputs
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    elif backend.startswith("llama"):
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
