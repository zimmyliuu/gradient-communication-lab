"""Read-only adapter to the specified frozen CPU implementation, not a rewrite."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import torch

REFERENCE_ROOT = Path('E:/dacomp/research/spl_model_communication')
sys.path.insert(0, str(REFERENCE_ROOT/'src'))
from splcomm.dynamiq import DynamiQAggregator, QuantizedPacket, lookup_table
from splcomm.packing import pack_unsigned, unpack_unsigned, tensor_bytes


def tensor_identity(value):
    return {'sha256':hashlib.sha256(tensor_bytes(value)).hexdigest(),
            'shape':list(value.shape), 'dtype':str(value.dtype), 'byte_order':'little',
            'bytes':value.numel()*value.element_size()}


def padded_length(length):
    if type(length) is not int or length<0:
        raise ValueError('Logical length must be a nonnegative integer')
    return (length+255)//256*256


def pad_input(value):
    if value.device.type!='cpu' or value.dtype!=torch.bfloat16 or value.ndim!=1 or not value.is_contiguous():
        raise ValueError('CPU fixture expects contiguous one-dimensional BF16')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite input')
    length=padded_length(value.numel())
    return torch.nn.functional.pad(value,(0,length-value.numel()))


def threshold_fixture(input_id, length):
    if length != padded_length(length):
        raise ValueError('Threshold length must be aligned to 256')
    seed=int.from_bytes(hashlib.sha256(('gpu_codec_v1:'+input_id).encode('utf-8')).digest()[:8],'little') % (1<<63)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    multiplier=torch.tensor(65535,dtype=torch.bfloat16)
    main=torch.rand((length//16,16),generator=generator,dtype=torch.float32).to(torch.bfloat16)*multiplier
    scale=torch.rand((length//256,16),generator=generator,dtype=torch.float32).to(torch.bfloat16)*multiplier
    return main,scale,{'generator':'torch.Generator(device=cpu)','seed':seed,'pytorch_version':torch.__version__,
        'order':['main','scale'],'generation':'FP32 rand -> BF16 -> multiply BF16(65535)',
        'multiplier_bf16_value':float(multiplier),'main':tensor_identity(main),'scale':tensor_identity(scale)}


def wire_size(length, bits):
    if bits not in (4,8) or length!=padded_length(length):
        raise ValueError('Unsupported fixed codec dimensions')
    return {'payload_bytes':(length*(bits+1)+7)//8,'scale_bytes':length//16+2*(length//256),
            'wire_bytes':(length*(bits+1)+7)//8+length//16+2*(length//256)}


class CpuReference:
    """Explicit thresholds avoid pool construction and any LCG progression."""
    def __init__(self, bits):
        if bits not in (4,8):
            raise ValueError('Only nominal b=4/8 are in this stage')
        self.bits=bits
        self.impl=DynamiQAggregator(world_size=1,target_bits=bits,profile='index_repaired_bf16',
            coupling='correlated',rounding_seed=0,max_bucket_numel=256,device='cpu',audit=True)

    def encode_native(self, x, main_thresholds, scale_thresholds):
        if x.device.type!='cpu' or x.dtype!=torch.bfloat16 or x.ndim!=1 or not x.is_contiguous():
            raise ValueError('Native input must be contiguous one-dimensional CPU BF16')
        length=x.numel()
        if length==0 or length%256:
            raise ValueError('Native codec requires positive 256-aligned length')
        for value,shape in ((main_thresholds,(length//16,16)),(scale_thresholds,(length//256,16))):
            if (value.device.type!='cpu' or value.dtype!=torch.bfloat16 or
                    tuple(value.shape)!=shape or not value.is_contiguous() or
                    not bool(torch.isfinite(value).all()) or bool(((value<0)|(value>65536)).any())):
                raise ValueError('Invalid explicit BF16 threshold tensor')
        self.impl.counters={}
        self.impl.last_audit=[]
        return self.impl.encode(x,self.bits,0,0,thresholds=(main_thresholds,scale_thresholds))

    def decode_native(self, packet):
        if packet.bits!=self.bits:
            raise ValueError('Packet bit width does not match adapter')
        return self.impl.decode(packet)

    def roundtrip(self, x, main_thresholds, scale_thresholds):
        padded=pad_input(x)
        if x.numel()==0:
            return {'logical_numel':0,'padded_numel':0,'packet':None,'wire':b'',
                    'decoded':x.clone(),'quantization_events':0,'audit':None}
        packet=self.encode_native(padded,main_thresholds,scale_thresholds)
        decoded=self.decode_native(packet)[:x.numel()].clone()
        return {'logical_numel':x.numel(),'padded_numel':padded.numel(),'packet':packet,
                'wire':packet.wire,'decoded':decoded,'quantization_events':1,'audit':self.impl.last_audit[-1]}
