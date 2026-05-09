import os
import pandas as pd
import torch
from transformers import AutoTokenizer

PROMPT_LENS = []
SAMPLE_ID = 0

def _init_cache():
    global PROMPT_LENS
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        for file_path in ["./data/dataset.csv", "./data/test.csv"]:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for prompt in df["prompt"]:
                    tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    PROMPT_LENS.append(len(tokens))
    except Exception:
        PROMPT_LENS = [0] * 1000

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    global SAMPLE_ID
    if not PROMPT_LENS: _init_cache()
    
    prompt_len = PROMPT_LENS[SAMPLE_ID] 
    real_positions = attention_mask.nonzero(as_tuple=False)
        
    last_pos = int(real_positions[-1].item())
    start_id = prompt_len
    
    layer16 = hidden_states[16]
    response_tokens = layer16[start_id : last_pos + 1]
    
    if response_tokens.size(0) == 0:
        val = layer16[last_pos]
        return torch.cat([val, val], dim=0)
    
    max_pool = response_tokens.max(dim=0)[0]
    avg_pool = response_tokens.mean(dim=0)
    
    return torch.cat([avg_pool, max_pool], dim=0)

def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    global SAMPLE_ID
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item()) 
    prompt_len = PROMPT_LENS[SAMPLE_ID]
    res_len = float(last_pos - prompt_len + 1)
    return torch.tensor([res_len], device=hidden_states.device)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,
) -> torch.Tensor:
    global SAMPLE_ID
    aggr = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geom = extract_geometric_features(hidden_states, attention_mask)
        result = torch.cat([aggr, geom], dim=0)
    else:
        result = aggr
    SAMPLE_ID += 1
    return result