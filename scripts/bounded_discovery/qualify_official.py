"""D0 only: official self-tests and small, preserved qualification cases."""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, runpy, subprocess, sys, traceback
from datetime import datetime
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/bounded_discovery_v1/bounded_discovery_v1_20261104_001'
sys.path.insert(0, str(ROOT / 'src'))
from gradient_comm_lab.bounded_discovery import official_codec as codec
TEST = ROOT / 'upstream/bounded_discovery_v1/dynamiq/cuda_kernels/tests/test_scaling_quantization_functional.py'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def fresh_decode():
    """This process receives packet bytes and metadata only, never source/RNG."""
    bundle = json.loads((RUN/'logs/d0_packet_manifest.json').read_bytes())
    rows=[]
    for entry in bundle:
        wire=(RUN/entry['wire_path']).read_bytes()
        packet=codec.OfficialPacket.from_bytes(wire,logical_numel=entry['logical_numel'],bits=entry['bits'])
        decoded=codec.decode(packet)
        torch.cuda.synchronize()
        rows.append(dict(case_id=entry['case_id'],bits=entry['bits'],output_sha256=thash(decoded),
                         expected_sha256=entry['output_sha256'],passed=thash(decoded)==entry['output_sha256']))
    dump(RUN/'logs/d0_fresh_process_decode.json',dict(module=codec.build_identity(),rows=rows,passed=all(r['passed'] for r in rows)))
    assert all(r['passed'] for r in rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh-decode',action='store_true');args=parser.parse_args()
    if args.fresh_decode: return fresh_decode()
    (RUN/'d0_cases').mkdir(exist_ok=True)
    record=dict(started_at=datetime.now().astimezone().isoformat(),status='running',official_selftest=None,cases=[],boundary_checks=[])
    out=RUN/'logs/d0_execution.json'
    try:
        codec.extension()
        record['module']=codec.build_identity()
        official=runpy.run_path(str(TEST))
        transcript=io.StringIO()
        try:
            with contextlib.redirect_stdout(transcript): official['main']()
            record['official_selftest']=dict(status='pass',configurations=48,operations_per_configuration=['run_roundtrip (also decompress_add)','run_dec_comp'],test_path=str(TEST),test_sha256=digest(TEST),tolerances='Official unchanged per dtype and strategy')
        except Exception as e:
            record['official_selftest']=dict(status='fail',exception=repr(e),traceback=traceback.format_exc(),test_sha256=digest(TEST))
            raise
        finally:
            (RUN/'logs/d0_official_selftest_console.log').write_text(transcript.getvalue(),encoding='utf-8')
            dump(out,record)
        gen=torch.Generator().manual_seed(714103)
        cases=[('zero_native_min',torch.zeros(256,dtype=torch.bfloat16)),
               ('constant',torch.full((256,),0.25,dtype=torch.bfloat16)),
               ('mixed_signed',torch.linspace(-2,2,512).to(torch.bfloat16)),
               ('sparse_large',torch.zeros(256,dtype=torch.bfloat16)),
               ('random',torch.randn(768,generator=gen).to(torch.bfloat16)),
               ('tail257',torch.randn(257,generator=gen).to(torch.bfloat16))]
        cases[3][1][127]=128
        manifest=json.loads((ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001/input_manifest.json').read_bytes())
        real=next(r for r in manifest['inputs'] if r.get('training_seed')==1101 and r.get('update_step')==100 and r.get('rank')==0)
        fixture=Path(real['fixture_path']);assert digest(fixture)==real['fixture_sha256']
        # Trusted, hash-verified local sealed fixture includes TorchVersion metadata.
        real_input=torch.load(fixture,weights_only=False)['input_logical']
        assert thash(real_input)==real['input_sha256'] and real_input.numel()==818048
        record['real_fixture']=dict(path=str(fixture),sha256=digest(fixture),input_sha256=thash(real_input),input_id=real['input_id'],transform='Use existing logical BF16 values; no second division, no inherited padding')
        cases.append(('real_seed1101_update100_rank0',real_input))
        packet_entries=[]
        for case_id,cpu in cases:
            torch.save(dict(input=cpu),RUN/'d0_cases'/f'{case_id}.pt')
            value=cpu.cuda();before=value.clone()
            for bits in (4,8):
                seed=714200+bits
                n=codec.padded_numel(value.numel())
                pool=codec.make_random_pool(n,seed,value.device);pool_before=pool.clone()
                packet=codec.encode(value,bits,rand_pool=pool)
                decoded=codec.decode(packet)
                again=codec.encode(value,bits,rand_pool=pool)
                torch.cuda.synchronize()
                row=dict(case_id=case_id,bits=bits,seed=seed,logical_numel=value.numel(),padded_numel=n,
                         rand_pool_sha256=thash(pool),rand_pool_exact_one_count=int((pool==1).sum()),
                         input_sha256=thash(value),output_sha256=thash(decoded),packet_sha256=thash(packet.data),
                         accounting=packet.accounting(),input_unchanged=torch.equal(before,value),random_pool_unchanged=torch.equal(pool,pool_before),
                         same_state_identical_packet=torch.equal(packet.data,again.data),output_finite=bool(torch.isfinite(decoded).all()),
                         nonzero_count=int(torch.count_nonzero(decoded)),max_abs_error=float((decoded.float()-value.float()).abs().max()),
                         squared_error=float(((decoded.double()-value.double())**2).sum()))
                if value.numel()<=768:
                    source=torch.zeros(n,dtype=value.dtype,device=value.device);source[:value.numel()]=value
                    ref=official['reference_decompressed_for_dtype'](source,pool,bits,16,3,torch.bfloat16)
                    expected=torch.as_tensor(ref[:value.numel()],dtype=torch.float32,device=value.device)
                    official['assert_close'](case_id,decoded,expected,3,torch.bfloat16)
                    row['official_reference_agreement']=True
                    row['reference_max_abs_difference']=float((decoded.float()-expected).abs().max())
                else:
                    row['official_reference_agreement']=None
                    row['reference_note']='Full real vector independently decoded from stored packet in fresh process; official CPU formula not re-executed at this size'
                assert row['input_unchanged'] and row['random_pool_unchanged'] and row['same_state_identical_packet'] and row['output_finite']
                wire_path=f'd0_cases/{case_id}_b{bits}.bin';wire=packet.to_bytes();(RUN/wire_path).write_bytes(wire)
                assert len(wire)==row['accounting']['packet_total_bytes']==row['accounting']['payload_bytes']+row['accounting']['scale_bytes']
                record['cases'].append(row)
                packet_entries.append(dict(case_id=case_id,bits=bits,logical_numel=value.numel(),wire_path=wire_path,wire_sha256=hashlib.sha256(wire).hexdigest(),output_sha256=thash(decoded)))
                print(json.dumps(dict(case_id=case_id,bits=bits,status='pass',bytes=len(wire))),flush=True)
                dump(out,record)
        # Native host metadata rejection; deliberately do not launch malformed kernels.
        for n in (0,257):
            try:
                codec.extension().scaling_compress(torch.zeros(512,dtype=torch.bfloat16,device='cuda'),torch.empty(600,dtype=torch.uint8,device='cuda'),torch.zeros(512,dtype=torch.bfloat16,device='cuda'),n,4,16,torch.cuda.current_stream().cuda_stream,3)
            except RuntimeError as e:
                record['boundary_checks'].append(dict(name=f'native_n{n}_rejected',passed=True,exception_type=type(e).__name__,message=str(e)))
            else: raise AssertionError('Native illegal length unexpectedly accepted')
        # These endpoint probes are diagnostic records, not a redefinition of legal RNG.
        for bits in (4,8):
            for endpoint in (0.,1.):
                value=torch.zeros(256,dtype=torch.bfloat16,device='cuda');pool=torch.full_like(value,endpoint)
                packet=codec.encode(value,bits,rand_pool=pool);decoded=codec.decode(packet);torch.cuda.synchronize()
                name=f'zero_rand_endpoint{int(endpoint)}_b{bits}'
                torch.save(dict(input=value.cpu(),random_pool=pool.cpu(),packet=packet.data.cpu(),decoded=decoded.cpu()),RUN/'d0_cases'/f'{name}.pt')
                expected=official['reference_decompressed_for_dtype'](value,pool,bits,16,3,torch.bfloat16)
                official['assert_close'](name,decoded,expected,3,torch.bfloat16)
                record['boundary_checks'].append(dict(name=name,passed=True,reference_agreement=True,nonzero_count=int(torch.count_nonzero(decoded)),max_abs_output=float(decoded.abs().max()),output_sha256=thash(decoded),packet_sha256=thash(packet.data)))
        dump(RUN/'logs/d0_packet_manifest.json',packet_entries)
        result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--fresh-decode'],cwd=ROOT,capture_output=True,timeout=120)
        (RUN/'logs/d0_fresh_process_console.log').write_bytes(result.stdout+b'\nSTDERR\n'+result.stderr)
        assert result.returncode==0, result.stderr.decode('utf-8',errors='replace')
        record['independent_decode']=json.loads((RUN/'logs/d0_fresh_process_decode.json').read_bytes())
        record['status']='pass'
    except Exception as error:
        record.update(status='fail',exception=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat();dump(out,record)
if __name__=='__main__':main()
