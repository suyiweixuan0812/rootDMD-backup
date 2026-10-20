"""Assert common training parameters match DMD; preserve all actual differences."""
import argparse
import json
from pathlib import Path
import hashlib


def flatten(d,prefix=''):
    result={}
    for k,v in d.items():
        key=prefix+'.'+k if prefix else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else:result[key]=v
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a,b=[json.loads(x.read_text()) for x in [args.baseline,args.candidate]]
    critical=['pretrained','resolution','mixed_precision','allow_tf32','seed','sample','train','dmd','ac_dmd','dataset','lora','num_epochs']
    for key in critical:assert a[key]==b[key],f'Unexpected difference in {key}'
    af,bf=flatten(a),flatten(b)
    differences={k:{'DMD':af.get(k),'PathIS':bf.get(k)} for k in sorted(af.keys()|bf.keys()) if af.get(k)!=bf.get(k)}
    result={'common_training_groups_exactly_equal':critical,'differences':differences,
       'baseline_config':str(args.baseline),'candidate_config':str(args.candidate),
       'config_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [args.baseline,args.candidate]},
       'important_method_difference':'Proposed method has zero fake updates and a different posterior/gradient surrogate; equality of external settings is not equality of algorithms or GPU time.',
       'evaluation_difference':'Development generation uses DrawBench64; final evaluation separately uses original PickScore2048 with the same frozen inference protocol.'}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Common groups exactly equal:',','.join(critical));print('All differences saved:',args.output)


if __name__=='__main__':main()
