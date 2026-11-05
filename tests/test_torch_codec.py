"""Focused G1a tests; never invoke the sealed G0 fixture-writing test suite."""
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gradient_comm_lab import torch_codec as tc
from gradient_comm_lab.cpu_reference import CpuReference,lookup_table,threshold_fixture

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='G1a requires CUDA')


def bytes_of(x):
    return bytes(x.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())


def scalar_pack(codes,width):
    """Independent Python integer oracle for low-bit-first ordering."""
    word=sum(int(value) << (width*index) for index,value in enumerate(codes))
    return word.to_bytes((len(codes)*width+7)//8,'little')


@pytest.mark.parametrize('bits',(4,8))
@pytest.mark.parametrize('count',(0,1,3,17,257,513))
def test_all_legal_pack_and_canonical_tail(bits,count):
    values=[i%((1<<bits)+1) for i in range(count)]
    codes=torch.tensor(values,device='cuda',dtype=torch.int64)
    payload=tc.pack_unsigned_cuda(codes,bits+1,max_code=1<<bits)
    assert bytes_of(payload)==scalar_pack(values,bits+1)
    restored=tc.unpack_unsigned_cuda(payload,bits+1,count,max_code=1<<bits)
    assert restored.cpu().tolist()==values
    if count*(bits+1)%8:
        corrupt=payload.clone();corrupt[-1]|=128
        with pytest.raises(ValueError,match='padding'):
            tc.unpack_unsigned_cuda(corrupt,bits+1,count,max_code=1<<bits)


@pytest.mark.parametrize('bits',(4,8))
@pytest.mark.parametrize('length',(0,1,15,16,255,256,257,1023,1024,1025))
def test_adapter_and_allocating_api_exact_reference(bits,length):
    x=torch.linspace(-1,1,length).to(torch.bfloat16)
    main,scale,_=threshold_fixture(f'pytest_adapter_{length}',(length+255)//256*256)
    expected=CpuReference(bits).roundtrip(x,main,scale)
    gx,gm,gs,lut=x.cuda(),main.cuda(),scale.cuda(),lookup_table(bits).cuda()
    before=[bytes_of(t) for t in (gx,gm,gs,lut)]
    packet,out=tc.roundtrip(gx,bits,gm,gs,lut,checked=True)
    assert bytes_of(packet.serialized())==expected['wire']
    assert bytes_of(out)==bytes_of(expected['decoded'])
    assert before==[bytes_of(t) for t in (gx,gm,gs,lut)]
    plan=tc.TorchCodecPlan(length,bits)
    plan.roundtrip_into(gx,gm,gs,lut,checked=True)
    assert plan.last_quantization_events==(1 if length else 0)
    assert plan.output.numel()==length
    if not length:assert packet.wire_bytes==0


@pytest.mark.parametrize('bits',(4,8))
def test_qscale_zero_preserves_negative_zero(bits):
    n=256
    signed=torch.tensor([-1,0,1,-(1<<(bits-1)),1<<(bits-1)],device='cuda',dtype=torch.int64).repeat(52)[:n]
    unsigned=signed+(1<<(bits-1))
    packet=tc.DevicePacket(tc.pack_unsigned_cuda(unsigned,bits+1,max_code=1<<bits),
        torch.zeros(16,device='cuda',dtype=torch.uint8),torch.ones(1,device='cuda',dtype=torch.bfloat16),n,n,bits)
    result=tc.decode(packet,lookup_table(bits).cuda(),checked=True)
    assert torch.equal(result.cpu().view(torch.int16),(signed.cpu()<0).to(torch.int16)*(-32768))


@pytest.mark.parametrize('bits',(4,8))
def test_preallocated_repeatability_and_checked_alias_rejection(bits):
    n=256;x=torch.linspace(-1,1,n).to(torch.bfloat16).cuda()
    main,scale,_=threshold_fixture('pytest_preallocation',n)
    main,scale,lut=main.cuda(),scale.cuda(),lookup_table(bits).cuda()
    plan=tc.TorchCodecPlan(n,bits)
    addresses=(plan.packet.payload.data_ptr(),plan.packet.scale_codes.data_ptr(),plan.packet.top_scales.data_ptr(),plan.output.data_ptr())
    first=bytes_of(plan.roundtrip_into(x,main,scale,lut,checked=True));wire=bytes_of(plan.packet.serialized())
    second=bytes_of(plan.roundtrip_into(x,main,scale,lut,checked=False))
    assert first==second and wire==bytes_of(plan.packet.serialized())
    assert addresses==(plan.packet.payload.data_ptr(),plan.packet.scale_codes.data_ptr(),plan.packet.top_scales.data_ptr(),plan.output.data_ptr())
    with pytest.raises(ValueError,match='alias'):
        plan.encode_into(plan.output,main,scale,lut,checked=True)
    invalid_payload=plan.decoded_padded.view(torch.uint8)[:plan.packet.payload.numel()]
    invalid=tc.DevicePacket(invalid_payload,plan.packet.scale_codes,plan.packet.top_scales,n,n,bits)
    with pytest.raises(ValueError,match='alias'):
        plan.decode_into(invalid,lut,checked=True)


@pytest.mark.parametrize('bits',(4,8))
def test_finite_top_scale_can_overflow_intermediate(bits):
    n=256
    payload=tc.pack_unsigned_cuda(torch.full((n,),1<<(bits-1),device='cuda',dtype=torch.int64),bits+1,max_code=1<<bits)
    packet=tc.DevicePacket(payload,torch.full((16,),255,device='cuda',dtype=torch.uint8),
        torch.full((1,),1e38,device='cuda',dtype=torch.bfloat16),n,n,bits)
    with pytest.raises(ValueError,match='Nonfinite decoded scales'):
        tc.decode(packet,lookup_table(bits).cuda(),checked=True)
