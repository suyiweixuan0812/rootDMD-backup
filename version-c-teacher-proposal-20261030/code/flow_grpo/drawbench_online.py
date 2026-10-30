"""Score saved DrawBench PNGs with the report's exact four-score conventions."""
import gc, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from flow_grpo.rewards import clip_score,aesthetic_score,pickscore_score,hpsv2_score

FACTORIES={'clipscore':clip_score,'aesthetic':aesthetic_score,'pickscore':pickscore_score,'hpsv2':hpsv2_score}

@torch.no_grad()
def score_saved_images(folder, records, device):
    scores=defaultdict(list)
    rows=[dict(index=i,**record,scores={}) for i,record in records]
    for name,factory in FACTORIES.items():
        if name == 'hpsv2':
            from .headless_hps import prepare_hps
            prepare_hps()
        scorer=factory(device)
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];pixels=[]
            for r in batch:
                with Image.open(folder/f'{r["index"]:06d}.png') as im:
                    pixels.append(np.array(im.convert('RGB'),copy=True))
            images=torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).to(device).float()/255
            values,_=scorer(images,[r['prompt'] for r in batch],[r.get('metadata',{}) for r in batch])
            if torch.is_tensor(values):values=values.detach().float().cpu().numpy()
            values=np.asarray(values).reshape(-1)
            assert len(values)==len(batch) and np.isfinite(values).all()
            for r,v in zip(batch,values):r['scores'][name]=float(v)
            scores[name].extend(values.tolist())
        del scorer
        gc.collect();torch.cuda.empty_cache()
    (folder/'scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return scores

def write_step_report(output_dir, step, config, counts, values, world):
    folder=output_dir/'eval'/f'step-{step}'
    expected=min(config.eval.final_num_prompts if step==config.num_epochs else config.eval.num_prompts,
                 len((Path(config.eval.dataset)/'test.txt').read_text().splitlines()))
    rows=[]
    for rank in range(world):rows.extend(json.loads((folder/f'rank-{rank}'/'scores.json').read_text()))
    rows.sort(key=lambda r:r['index'])
    assert [r['index'] for r in rows]==list(range(expected))
    means={k:float(np.mean([r['scores'][k] for r in rows],dtype=np.float64)) for k in FACTORIES}
    for k in FACTORIES:
        assert counts[k]==expected
        assert np.isclose(means[k],values['eval/'+k],atol=1e-10,rtol=1e-10)
    table={'CLIP Score':means['clipscore']*100,'Aesthetic Score':means['aesthetic'],
           'Pick Score':means['pickscore']*26,'HP Score':means['hpsv2']*100}
    report={'method':config.method_name,'dataset':'DrawBench','step':step,'num_images':expected,
            'table':table,'raw_repository_means':means,'protocol':{
              'seed':config.eval.seed,'world_size':config.execution.physical_world_size,'logical_evaluation_streams':world,'generation_batch_per_rank':config.sample.test_batch_size,
              'scoring_batch_per_rank':8,'resolution':config.resolution,'nfe':4,'ema':bool(step),
              'evaluation_dataset':'DrawBench','eval_freq':config.eval_freq,
              'prompt_file_sha256':hashlib.sha256((Path(config.eval.dataset)/'test.txt').read_bytes()).hexdigest(),
              'unique_prompts':len({r['prompt'] for r in rows}),
              'checkpoint_role':'saved EMA generator LoRA' if step else 'initial zero-output LoRA',
              'scoring_input':'identical saved PNGs for all four models'}}
    (folder/'per-image-scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (folder/'final-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('DRAWBENCH_STEP_RESULT '+json.dumps(report),flush=True)
