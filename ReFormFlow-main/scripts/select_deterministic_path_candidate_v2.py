"""Apply the predeclared Dev64 rule once; never inspect test/holdout scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CASES = ['dmd', 'dmd_fp32', 'base', 'time', 'paired', 'paired_time',
         'bridge', 'bridge_time', 'adaptive_bridge', 'adaptive_bridge_time', 'last', 'last_time']
METRICS = ['CLIP Score', 'Aesthetic Score', 'Pick Score', 'HP Score']
CONTROLS = {'dmd', 'dmd_fp32', 'paired', 'paired_time'}


def select(root):
    tables, hashes = {}, {}
    for case in CASES:
        folder = root / 'evidence' / f'sd3-det-{case}-seed42'
        assert (folder / 'status.txt').read_text().strip() == 'COMPLETED_DETERMINISTIC_DEV64', case
        path = folder / 'final-metrics.json'
        report = json.loads(path.read_text())
        assert report['num_images'] == 64
        tables[case] = report['table']
        hashes[case] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = {k: max(tables[c][k] for c in ['dmd', 'dmd_fp32']) for k in METRICS}
    rows = []
    for case in CASES:
        delta = {k: tables[case][k] - reference[k] for k in METRICS}
        eligible = (case not in CONTROLS and delta['Pick Score'] > 0 and delta['HP Score'] > 0
                    and delta['CLIP Score'] >= -.10 and delta['Aesthetic Score'] >= -.03)
        rank = [min(delta['Pick Score'] / .2, delta['HP Score'] / .5),
                delta['Pick Score'] / .2 + delta['HP Score'] / .5]
        rows.append(dict(case=case, table=tables[case], delta_vs_larger_baseline=delta,
                         eligible=eligible, selection_rank=rank, is_control=case in CONTROLS))
    qualifying = [r for r in rows if r['eligible']]
    pool = qualifying or [r for r in rows if not r['is_control']]
    chosen = max(pool, key=lambda r: r['selection_rank'])
    faithful = [r for r in rows if r['case'] in {'base', 'time', 'last', 'last_time'}]
    faithful_pool = [r for r in faithful if r['eligible']] or faithful
    faithful_choice = max(faithful_pool, key=lambda r: r['selection_rank'])
    return dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                status='QUALIFIED_DEV_CANDIDATE' if qualifying else 'NO_QUALIFIED_DEV_CANDIDATE',
                interpretation=('The selected candidate requires independent test validation.' if qualifying else
                                'No path method met all predeclared guards. The best-ranked path method is evaluated for diagnosis only.'),
                selected_case=chosen['case'], paired_control='paired_time' if chosen['case'].endswith('time') else 'paired',
                pure_teacher_alternative=dict(case=faithful_choice['case'], eligible=faithful_choice['eligible'],
                    paired_control='paired_time' if faithful_choice['case'].endswith('time') else 'paired',
                    role='Same predeclared ranking restricted to pure Teacher mappings; additional fidelity comparison if primary selected proposal is bridged; never replaces primary selection by test results'),
                reference=reference, trials=rows, dev_report_sha256=hashes,
                selection_inputs='Only the twelve predeclared deterministic seed42 DrawBench64 reports; no PickScore2048/holdout138 scores.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round-root', type=Path, default=Path(__file__).resolve().parents[1] / 'logs/path-ratio-round1')
    a = p.parse_args()
    out = a.round_root / 'deterministic-selection.json'
    assert not out.exists(), 'Never overwrite frozen selection'
    result = select(a.round_root)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
