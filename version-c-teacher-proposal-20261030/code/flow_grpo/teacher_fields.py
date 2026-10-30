import torch

def make_fields(transformer,embeds,pooled,negative,negative_pooled,total_steps,
                proposal_cfg=1.,mixed_precision_dtype=torch.bfloat16,micro_batch=1):
    def predict(sample,s,teacher=False):
        if sample.shape[0]>micro_batch:
            return torch.cat([predict(part,s,teacher) for part in sample.split(micro_batch)])
        n=sample.shape[0]
        cond,pool=embeds.expand(n,-1,-1),pooled.expand(n,-1)
        guided=teacher and proposal_cfg>1
        if guided:
            hidden=torch.cat([sample,sample])
            cond=torch.cat([negative.expand(n,-1,-1),cond])
            pool=torch.cat([negative_pooled.expand(n,-1),pool])
        else:hidden=sample
        context=transformer.disable_adapter() if teacher else torch.autocast(sample.device.type,enabled=False)
        with context,torch.autocast(sample.device.type,dtype=mixed_precision_dtype,
                                  enabled=mixed_precision_dtype is not None):
            result=transformer(hidden_states=hidden,
                timestep=torch.full((hidden.shape[0],),s*total_steps,device=sample.device),
                encoder_hidden_states=cond,pooled_projections=pool,return_dict=False)[0]
        if guided:
            uncond,cond=result.chunk(2)
            result=uncond+proposal_cfg*(cond-uncond)
        return result
    return lambda x,s:predict(x,s,False),lambda x,s:predict(x,s,True)
