"""Load inference LoRA without PEFT0.10 silently copying FP32 into BF16 params."""
from pathlib import Path
import hashlib
import torch
from peft import PeftConfig,get_peft_model,get_peft_model_state_dict
from safetensors.torch import load_file


def load_exact_fp32_lora(base_model,adapter_path,adapter_name='student'):
    folder=Path(adapter_path)
    config=PeftConfig.from_pretrained(str(folder),local_files_only=True)
    model=get_peft_model(base_model,config,adapter_name=adapter_name)
    for name,p in model.named_parameters():
        if 'lora_' in name:p.data=p.data.float()
    model.load_adapter(str(folder),adapter_name=adapter_name,is_trainable=False,local_files_only=True)
    model.set_adapter(adapter_name)
    original=load_file(str(folder/'adapter_model.safetensors'),device='cpu')
    loaded=get_peft_model_state_dict(model,adapter_name=adapter_name)
    assert original.keys()==loaded.keys()
    exact=sum(torch.equal(original[k],loaded[k].detach().cpu()) and original[k].dtype==loaded[k].dtype for k in original)
    assert exact==len(original),'Inference adapter was rounded during loading'
    with (folder/'adapter_model.safetensors').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    record={'source':str(folder),'source_sha256':sha,'tensor_count':len(original),
            'all_loaded_tensors_and_dtypes_exact':True,
            'dtype':sorted({str(v.dtype) for v in loaded.values()})}
    model.requires_grad_(False).eval()
    return model,record
