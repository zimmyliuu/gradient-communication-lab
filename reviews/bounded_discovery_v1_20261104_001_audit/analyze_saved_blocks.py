"""Read-only postprocessing of saved D1 inputs/outputs; no codec or GPU calls."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    assert root not in args.output.resolve().parents
    torch.set_num_threads(2)
    raw_path = root / 'metrics_raw.jsonl'
    raw = [json.loads(s) for s in raw_path.read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((root / 'input_manifest.json').read_text(encoding='utf-8'))
    methods = ['official_dynamiq_fixed_codec_b8', 'uniform_g128_q8']
    result = dict(scope='posthoc_saved_outputs_only_no_new_quantization',
                  raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  block_size=256, method_order=methods, snapshots=[])
    for snapshot in sorted({i['snapshot_id'] for i in manifest['inputs']}):
        items = sorted([i for i in manifest['inputs'] if i['snapshot_id'] == snapshot], key=lambda i: i['worker_id'])
        energy = None
        squared = {}
        zeros = {}
        workers = []
        for item in items:
            fixture = torch.load(item['fixture_path'], map_location='cpu', weights_only=False)
            x = fixture['input_logical'].to(torch.float64)
            n = x.numel()
            p = ((n + 255) // 256) * 256
            def blocks(t):
                return torch.nn.functional.pad(t, (0, p-n)).reshape(-1, 256).sum(1)
            en = blocks(x.square())
            energy = en.clone() if energy is None else energy + en
            worker = dict(worker_id=item['worker_id'], zero_fraction=float((x == 0).double().mean()), methods={})
            for method in methods:
                rows = sorted([r for r in raw if r['record_type'] == 'local' and r['input_id'] == item['input_id'] and r['method_id'] == method], key=lambda r:r['repeat_id'])
                assert len(rows) == 8 and {r['repeat_id'] for r in rows} == set(range(8))
                terms, zero_sse, all_sse = [], [], []
                for row in rows:
                    y = torch.load(root / row['decoded_output_path'], map_location='cpu', weights_only=True).to(torch.float64)
                    se = (y-x).square()
                    sse = float(se.sum())
                    assert math.isclose(sse, row['squared_error'], rel_tol=1e-12, abs_tol=1e-24)
                    terms.append(blocks(se))
                    zero_sse.append(float(se[x == 0].sum()))
                    all_sse.append(sse)
                mean_blocks = torch.stack(terms).mean(0)
                squared[method] = mean_blocks.clone() if method not in squared else squared[method] + mean_blocks
                zeros[method] = zeros.get(method, 0.0) + sum(zero_sse)/8
                worker['methods'][method] = dict(mean_sse=sum(all_sse)/8, min_sse=min(all_sse), max_sse=max(all_sse))
            worker['uniform_to_official_sse_ratio'] = worker['methods'][methods[1]]['mean_sse']/worker['methods'][methods[0]]['mean_sse']
            workers.append(worker)
        delta = squared[methods[0]] - squared[methods[1]]
        pos, neg = delta.clamp_min(0), (-delta).clamp_min(0)
        totals = {m: float(squared[m].sum()) for m in methods}
        def contributors(values):
            ids = torch.argsort(values, descending=True, stable=True)
            total = float(values.sum())
            counts = [1, 10, math.ceil(values.numel()*0.01), math.ceil(values.numel()*0.1)]
            return dict(total=total, top_mass_shares={str(k): float(values[ids[:k]].sum())/total if total else None for k in counts},
                        top_blocks=[dict(block_id=int(j), first_coordinate=int(j)*256, delta_official_minus_uniform=float(delta[j]),
                                         input_energy=float(energy[j]), official_sse=float(squared[methods[0]][j]), uniform_sse=float(squared[methods[1]][j])) for j in ids[:10] if values[j] > 0])
        top_energy = torch.argsort(energy, descending=True, stable=True)[:math.ceil(energy.numel()*0.01)]
        result['snapshots'].append(dict(snapshot_id=snapshot, worker_count=len(items), workers=workers,
            blocks=int(delta.numel()), source_energy=float(energy.sum()), mean_local_sse_sum=totals,
            zero_input_sse_share={m: zeros[m]/totals[m] for m in methods},
            delta_total=float(delta.sum()), positive_block_fraction=float((delta > 0).double().mean()),
            official_worse_contributors=contributors(pos), uniform_worse_contributors=contributors(neg),
            top_1pct_energy_blocks=dict(count=int(top_energy.numel()), input_energy_share=float(energy[top_energy].sum()/energy.sum()),
                official_sse_share=float(squared[methods[0]][top_energy].sum())/totals[methods[0]],
                uniform_sse_share=float(squared[methods[1]][top_energy].sum())/totals[methods[1]],
                delta=float(delta[top_energy].sum()))))
    result['limitations'] = ['Exploratory decomposition on development inputs, not independent confirmation.',
        'Blocks are shared coordinate ranges of 256; padding contributes zero to the cropped error.',
        'A block error difference compares averages of independent RNG runs, not matched random draws.',
        'Error concentration or input energy association is not a causal mechanism.',
        'Exact-zero input attribution does not cover nonzero integer-lattice RNG endpoints.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for s in result['snapshots']:
        print(json.dumps({k:s[k] for k in ['snapshot_id','mean_local_sse_sum','zero_input_sse_share','delta_total','positive_block_fraction','top_1pct_energy_blocks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
