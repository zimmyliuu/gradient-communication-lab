"""CPU-only G0 semantic probes; this is not GPU parity or G1 acceptance."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from gradient_comm_lab.cpu_reference import (
    CpuReference, QuantizedPacket, REFERENCE_ROOT, pack_unsigned,
    tensor_bytes, tensor_identity, threshold_fixture, wire_size,
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exception_of(function):
    try:
        function()
    except Exception as exc:
        return {'exception': type(exc).__name__, 'message': str(exc)}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_relative_to((ROOT/'results/gpu_codec_v1').resolve()):
        raise ValueError('Probe output must remain in the new project run directory')
    output = run_dir/'logs/cpu_reference_semantics.json'
    if output.exists():
        raise FileExistsError('Preserve prior probe output; choose a fresh run directory')
    torch.set_num_threads(4)
    rng_before = torch.get_rng_state().clone()
    epsilon = torch.tensor(1e-23, dtype=torch.bfloat16)
    multiplier = torch.tensor(65535., dtype=torch.bfloat16)
    record = {
        'schema_version': 'gpu_codec_v1.0', 'template_only': False,
        'run_id': run_dir.name, 'created_at': datetime.now().astimezone().isoformat(),
        'scope': 'G0_CPU_reference_semantics_only_no_GPU_implementation_or_G1_acceptance',
        'gpu_parity_tested': False, 'g1_acceptance_claimed': False,
        'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads(),
        'constants': {'BF16_65535': float(multiplier), 'BF16_epsilon_1e_minus_23': float(epsilon),
                      'epsilon_bf16_bits': int(epsilon.view(torch.int16))},
    }
    assert float(multiplier) == 65536.0

    # Enumerate bit patterns, rather than approximating a numerical grid.
    finite = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = finite[torch.isfinite(finite)]
    actual = finite + 1e-23
    explicit_bf16_scalar = finite + epsilon
    fp32_scalar_then_cast = (finite.float() + 1e-23).to(torch.bfloat16)
    bf16_differences = actual.view(torch.int16) != explicit_bf16_scalar.view(torch.int16)
    fp32_differences = actual.view(torch.int16) != fp32_scalar_then_cast.view(torch.int16)
    representatives = []
    for index in torch.nonzero(fp32_differences).reshape(-1)[:5].tolist():
        representatives.append({
            'input': float(finite[index]), 'input_bits': int(finite[index].view(torch.int16)),
            'reference_output': float(actual[index]),
            'reference_output_bits': int(actual[index].view(torch.int16)),
            'fp32_literal_add_then_cast_output': float(fp32_scalar_then_cast[index]),
            'fp32_literal_add_then_cast_output_bits': int(fp32_scalar_then_cast[index].view(torch.int16)),
        })
    record['epsilon_exhaustive_probe'] = {
        'finite_BF16_bit_patterns': finite.numel(),
        'reference_vs_explicit_BF16_scalar_mismatch_count': int(bf16_differences.sum()),
        'reference_vs_FP32_literal_scalar_then_cast_mismatch_count': int(fp32_differences.sum()),
        'representative_counterexamples': representatives,
        'interpretation': 'Retain BF16 scalar conversion and each operation rounding; literal FP32 epsilon addition is not equivalent.',
    }
    assert int(bf16_differences.sum()) == 0
    assert int(fp32_differences.sum()) == 1527

    record['zero_cases'], record['finite_overflow_boundaries'], record['adapter_probes'] = [], [], []
    fixture_id, logical_length, padded_length = 'g0_semantics_normal_257', 257, 512
    main_thresholds, scale_thresholds, identity = threshold_fixture(fixture_id, padded_length)
    seed = int.from_bytes(hashlib.sha256(('gpu_codec_v1:'+fixture_id).encode()).digest()[:8], 'little') % (1<<63)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    expected_main = torch.rand((padded_length//16,16), generator=generator, dtype=torch.float32).to(torch.bfloat16)*multiplier
    expected_scale = torch.rand((padded_length//256,16), generator=generator, dtype=torch.float32).to(torch.bfloat16)*multiplier
    assert seed == identity['seed']
    assert torch.equal(main_thresholds, expected_main) and torch.equal(scale_thresholds, expected_scale)
    repeated_main, repeated_scale, repeated_identity = threshold_fixture(fixture_id, padded_length)
    assert identity == repeated_identity
    assert torch.equal(main_thresholds,repeated_main) and torch.equal(scale_thresholds,repeated_scale)
    x = torch.randn(logical_length, generator=torch.Generator(device='cpu').manual_seed(7301)).to(torch.bfloat16)
    original = [value.clone() for value in (x, main_thresholds, scale_thresholds)]
    record['ordinary_threshold_fixture'] = {'input_id': fixture_id, **identity,
        'independent_generation_matches': True, 'repeat_generation_matches': True,
        'same_threshold_tensors_used_for_both_bit_widths': True}

    for bits in (4,8):
        reference = CpuReference(bits)
        initial_control = copy.deepcopy(reference.impl.state_dict())
        lut = reference.impl.tables[bits]
        for threshold in (0.,65536.):
            packet = reference.encode_native(torch.zeros(256,dtype=torch.bfloat16),
                torch.full((16,16),threshold,dtype=torch.bfloat16),
                torch.full((1,16),threshold,dtype=torch.bfloat16))
            codes, scales, scale_codes, top = reference.impl.decode_components(packet)
            decoded = reference.decode_native(packet)
            assert not bool(codes.any()) and not bool(decoded.view(torch.int16).any())
            assert scale_codes.unique().tolist() == ([128] if threshold == 0 else [127])
            assert top.view(torch.int16).tolist() == [6593]
            assert len(packet.wire) == wire_size(256,bits)['wire_bytes']
            record['zero_cases'].append({'nominal_bits':bits,'threshold_value':threshold,
                'code_values':codes.unique().tolist(),'scale_code_values':scale_codes.unique().tolist(),
                'top_scales':top.float().tolist(),'top_scale_bits':top.view(torch.int16).tolist(),
                'decoded_scales':scales.float().unique().tolist(),
                'decoded_output_bit_patterns':decoded.view(torch.int16).unique().tolist(),
                **wire_size(256,bits), 'packet_sha256':hashlib.sha256(packet.wire).hexdigest()})

        outcome = reference.roundtrip(x,main_thresholds,scale_thresholds)
        duplicate = reference.roundtrip(x,main_thresholds,scale_thresholds)
        assert outcome['wire'] == duplicate['wire']
        assert torch.equal(outcome['decoded'].view(torch.int16),duplicate['decoded'].view(torch.int16))
        assert outcome['logical_numel']==logical_length and outcome['padded_numel']==padded_length
        assert torch.equal(outcome['audit']['input'][:logical_length],x)
        assert not bool(outcome['audit']['input'][logical_length:].any())
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(original,(x,main_thresholds,scale_thresholds)))
        assert initial_control == reference.impl.state_dict()
        empty_thresholds = threshold_fixture('g0_empty',0)
        counters_before = copy.deepcopy(reference.impl.counters)
        empty = reference.roundtrip(torch.empty(0,dtype=torch.bfloat16),*empty_thresholds[:2])
        assert empty['packet'] is None and empty['wire']==b'' and empty['quantization_events']==0
        assert counters_before==reference.impl.counters and initial_control==reference.impl.state_dict()
        record['adapter_probes'].append({'nominal_bits':bits,'lut':tensor_identity(lut),
            'positive_normalization_endpoint':float(lut[(1<<bits)+(1<<(bits-1))]),
            'normal_fixture_input':tensor_identity(x),'logical_numel':logical_length,'padded_numel':padded_length,
            'packet_sha256':hashlib.sha256(outcome['wire']).hexdigest(),'wire_bytes':len(outcome['wire']),
            'decoded':tensor_identity(outcome['decoded']),'repeat_packet_and_output_identical':True,
            'input_and_thresholds_unchanged':True,'padding_is_zero_and_output_trimmed':True,
            'LCG_and_protocol_control_state_unchanged':True,'random_pool_count':len(reference.impl.pools),
            'empty_input_skips_native_codec':True})

        zero_main=torch.zeros((16,16),dtype=torch.bfloat16)
        zero_scale=torch.zeros((1,16),dtype=torch.bfloat16)
        maximal=torch.full((256,),torch.finfo(torch.bfloat16).max,dtype=torch.bfloat16)
        maximum_packet=reference.encode_native(maximal,zero_main,zero_scale)
        maximum_error=exception_of(lambda:reference.decode_native(maximum_packet))
        assert (maximum_error is not None)==(bits==4)
        large_top_packet=QuantizedPacket(
            pack_unsigned(torch.full((256,),1<<(bits-1),dtype=torch.int64),bits+1),
            bytes([255]*16)+tensor_bytes(torch.tensor([1e38],dtype=torch.bfloat16)),256,bits)
        large_top_error=exception_of(lambda:reference.decode_native(large_top_packet))
        assert large_top_error and large_top_error['exception']=='FloatingPointError'
        record['finite_overflow_boundaries'].append({'nominal_bits':bits,
            'maximum_finite_input_encode':'pass','maximum_finite_input_decode_error':maximum_error,
            'finite_positive_top_with_zero_codes_and_qscale255_decode_error':large_top_error,
            'interpretation':'Finite inputs/top scales are necessary but do not guarantee finite intermediate BF16 decode; preserve reference rejection, never silently reorder multiplication/division.'})

    assert torch.equal(rng_before,torch.get_rng_state())
    record['global_torch_CPU_RNG_unchanged']=True
    source_paths=[ROOT/'scripts/probe_reference_semantics.py',ROOT/'src/gradient_comm_lab/cpu_reference.py',
                  REFERENCE_ROOT/'src/splcomm/dynamiq.py',REFERENCE_ROOT/'src/splcomm/packing.py']
    record['source_sha256']={p.as_posix():file_hash(p) for p in source_paths}
    record['status']='passed_CPU_semantic_probes_only'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':record['status'],'output':str(output),
        'epsilon_scalar_mismatch_count':int(fp32_differences.sum()),'gpu_parity_tested':False}))


if __name__=='__main__':
    main()
