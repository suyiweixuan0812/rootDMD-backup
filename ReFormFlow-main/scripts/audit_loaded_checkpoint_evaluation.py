"""Attach the checkpoint's TRUE training settings to an evaluation-only report.

The evaluation launcher inherits an unused sampler configuration; those values
must not be reported as the training history of an arbitrary loaded checkpoint.
No images, per-image scores or table values are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    run=args.run_dir.resolve()
    project=Path(__file__).resolve().parents[1]
    config=json.loads((run/'config.json').read_text())
    checkpoint=(project/config['lora']['generator_path']).resolve()
    training_config=checkpoint.parents[1]/'config.json'
    actual=json.loads(training_config.read_text())
    file=run/'eval/step-1/final-metrics.json'
    report=json.loads(file.read_text())
    assert report['protocol']['checkpoint_evaluation_only']['all_exported_tensors_exactly_equal']
    before=report['table'].copy()
    report['protocol']['path_is']=actual.get('path_is',{'enabled':False})
    report['protocol']['posterior_consistency']=actual.get('posterior_consistency',{'enabled':False})
    report['protocol']['posterior_time_weight']=actual.get('posterior_time_weight',{'mode':'none','effective_weight':1.0})
    report['protocol']['checkpoint_training_configuration']={
        'path':str(training_config),'sha256':hashlib.sha256(training_config.read_bytes()).hexdigest(),
        'method_name':actual.get('method_name','DMD'), 'seed':actual['seed'],
        'outer_steps':actual['num_epochs'],'generator_update_interval':actual['dmd']['fake_update_ratio'],
        'dataset':actual['dataset'],'sampler_fields_from_actual_training_configuration':True}
    assert before==report['table']
    file.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    evidence=project/'logs/path-ratio-round1/evidence'/run.name
    shutil.copy2(file,evidence/file.name)
    shutil.copy2(training_config,evidence/'checkpoint-training-config.json')
    print(run.name,'training metadata audited; score values unchanged')


if __name__=='__main__':main()
