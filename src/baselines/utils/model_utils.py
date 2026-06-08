from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from openai import OpenAI

USE_OPENAI_DEFAULT_MAX_OUTPUT_TOKENS = True

GPT54_LONG_CONTEXT_THRESHOLD = 272_000

_USAGE_TOTAL = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}


def reset_usage_totals():
    _USAGE_TOTAL["input_tokens"] = 0
    _USAGE_TOTAL["cached_input_tokens"] = 0
    _USAGE_TOTAL["output_tokens"] = 0
    _USAGE_TOTAL["total_tokens"] = 0
    _USAGE_TOTAL["cost_usd"] = 0.0


def get_total_usage():
    return dict(_USAGE_TOTAL)


def _accumulate_usage(usage_dict):
    _USAGE_TOTAL["input_tokens"] += int(usage_dict.get("input_tokens", 0) or 0)
    _USAGE_TOTAL["cached_input_tokens"] += int(usage_dict.get("cached_input_tokens", 0) or 0)
    _USAGE_TOTAL["output_tokens"] += int(usage_dict.get("output_tokens", 0) or 0)
    _USAGE_TOTAL["total_tokens"] += int(usage_dict.get("total_tokens", 0) or 0)
    _USAGE_TOTAL["cost_usd"] += float(usage_dict.get("cost_usd", 0.0) or 0.0)


def _get_attr(obj, name, default=0):
    return getattr(obj, name, default) if obj is not None else default


def _get_price_per_million(llm_type, input_tokens=0):
    if llm_type.lower().startswith("gpt-5.4"):
        price = {"input": 2.50, "cached_input": 0.25, "output": 15.00}
        if input_tokens > GPT54_LONG_CONTEXT_THRESHOLD:
            price = {
                "input": price["input"] * 2,
                "cached_input": price["cached_input"] * 2,
                "output": price["output"] * 1.5,
            }
        return price

    return {"input": 0.25, "cached_input": 0.025, "output": 2.00}


def _build_usage_dict(usage, llm_type):
    prompt_tokens = int(_get_attr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(_get_attr(usage, "output_tokens", 0) or 0)
    total_tokens = int(_get_attr(usage, "total_tokens", 0) or 0)
    input_details = _get_attr(usage, "input_tokens_details", None)
    output_details = _get_attr(usage, "output_tokens_details", None)
    cached_input_tokens = int(_get_attr(input_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(_get_attr(output_details, "reasoning_tokens", 0) or 0)
    billable_input_tokens = max(prompt_tokens - cached_input_tokens, 0)

    price_per_million = _get_price_per_million(llm_type, prompt_tokens)
    price = {key: value / 1_000_000 for key, value in price_per_million.items()}
    cost_usd = (
        billable_input_tokens * price["input"]
        + cached_input_tokens * price["cached_input"]
        + completion_tokens * price["output"]
    )

    return {
        "input_tokens": prompt_tokens,
        "billable_input_tokens": billable_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "price_per_million": price_per_million,
        "long_context_pricing": llm_type.lower().startswith("gpt-5.4") and prompt_tokens > GPT54_LONG_CONTEXT_THRESHOLD,
        "cost_usd": cost_usd,
    }


def load_model(llm_type):
    """
    huggingface_model: bool 
    llm_type: str - Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B, Qwen/Qwen3-4B-Instruct-2507, gpt-4.1-mini
    """

    if llm_type.lower().startswith("gpt-"):
        tokenizer = None
        model = None
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_type)
        model = AutoModelForCausalLM.from_pretrained(llm_type, 
                                                     torch_dtype=torch.float16, 
                                                     device_map='auto') 

    return tokenizer, model


def load_reranker(llm_type):
    if llm_type == "Qwen3-Reranker-0.6B":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", device_map='auto').eval()
    
    else:
        raise ValueError(f"Unsupported Hugging Face model type: {llm_type}")

    return tokenizer, model 


def Qwen_generate_(system_prompt, input_prompt, max_new_tokens, tokenizer, model, thinking_mode=False, do_sample=False, temperature=None, top_p=None, top_k=None):
    
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": input_prompt}
    ]

    model_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding = 'longest').to(model.device)
    
    _ = model.eval()
    with torch.no_grad():
        # Avoid storing per-step logits because they increase VRAM pressure.
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if isinstance(model_input, torch.Tensor):
            output = model.generate(model_input, **gen_kwargs)
            prompt_len = model_input.shape[-1]
        else:
            output = model.generate(**model_input, **gen_kwargs)
            prompt_len = model_input["input_ids"].shape[-1]

    if hasattr(output, "sequences"):
        seq = output.sequences[0]
    else:
        seq = output[0]
    result = seq[prompt_len:]
    response = tokenizer.decode(result, skip_special_tokens=True)
    return _, response


def gpt_generate(system_prompt, input_prompt, max_new_tokens, llm_type, do_sample=False, temperature=None, top_p=None):
    client = OpenAI()
    
    input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]
    
    api_kwargs = {
        "model": llm_type,
        "input": input,
        "reasoning": {"effort": "low"}
    }
    if llm_type.lower().startswith("gpt-5.4"):
        api_kwargs["reasoning"] = {"effort": "none"}
        api_kwargs["truncation"] = "disabled"
        if max_new_tokens is not None:
            api_kwargs["max_output_tokens"] = max_new_tokens
    elif not USE_OPENAI_DEFAULT_MAX_OUTPUT_TOKENS and max_new_tokens is not None:
        api_kwargs["max_output_tokens"] = max_new_tokens
    
    response = client.responses.create(**api_kwargs)
    content = response.output_text
    
    usage_dict = _build_usage_dict(response.usage, llm_type)
    usage_dict.update({
        "response_id": response.id,
        "response_status": response.status,
        "incomplete_reason": _get_attr(response.incomplete_details, "reason", None),
        "model": response.model,
        "max_output_tokens": _get_attr(response, "max_output_tokens", None),
        "truncation": _get_attr(response, "truncation", None),
        "reasoning_effort": _get_attr(_get_attr(response, "reasoning", None), "effort", None),
        "api_kwargs_summary": {
            "model": api_kwargs.get("model"),
            "reasoning": api_kwargs.get("reasoning"),
            "truncation": api_kwargs.get("truncation"),
            "max_output_tokens": api_kwargs.get("max_output_tokens"),
        },
    })
    _accumulate_usage(usage_dict)

    return usage_dict, content



def LLM(system_prompt, input_prompt, llm_type, max_new_tokens, tokenizer, model, do_sample=False, temperature=None, top_p=None, top_k=None):
    if 'Qwen' in llm_type:
        return Qwen_generate_(system_prompt, input_prompt, max_new_tokens, tokenizer, model, thinking_mode=False, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)

    if 'gpt' in llm_type.lower():
        return gpt_generate(system_prompt, input_prompt, max_new_tokens, llm_type, do_sample=do_sample, temperature=temperature, top_p=top_p)



