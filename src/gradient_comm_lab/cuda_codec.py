"""Fixed local b4/b8 custom CUDA codec. No collective, controller, or RNG work.

One CTA per 256 elements, one encode and one decode launch. No global packing
indices or quantization intermediates. Checked calls synchronize an explicit
native status word after metadata/value validation; trusted calls do neither.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from .torch_codec import DevicePacket,TorchCodecPlan,_cuda_device,_require_metadata,_require_device_condition

METHOD_ID='cuda_optimized_repaired'
ROOT=Path(__file__).resolve().parents[2]
_manifest_path=ROOT/'results/gpu_codec_v1/gpu_codec_v1_20261104_001/logs/cuda_build_manifest.json'
_module=None

def set_build_manifest(path):
    global _manifest_path,_module
    path=Path(path).resolve()
    if path!=_manifest_path:_module=None
    _manifest_path=path

def load_extension():
    global _module
    if _module is None:
        manifest=json.loads(_manifest_path.read_text(encoding='utf-8'))
        path=Path(manifest['module_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['module_sha256']:
            raise ValueError('CUDA build artifact hash mismatch')
        spec=importlib.util.spec_from_file_location(manifest['module_name'],path)
        _module=importlib.util.module_from_spec(spec);spec.loader.exec_module(_module)
    return _module

STATUS_MESSAGES=((1,'Nonfinite normalized input'),(2,'Search index out of range'),
 (4,'Nonfinite rounding probability'),(8,'Invalid top scale'),(16,'Invalid scale code'),
 (64,'Nonfinite or nonpositive top scale'),(32,'Unused packed code'),
 (128,'Nonfinite decoded scales'),(256,'Nonfinite decoded output'))

class CudaCodecPlan(TorchCodecPlan):
    def __init__(self,logical_numel,bits,device='cuda'):
        if type(logical_numel) is not int or logical_numel<0 or type(bits) is not int or bits not in (4,8):
            raise ValueError('Expected nonnegative logical length and nominal b4/b8')
        if logical_numel>2147483392:raise ValueError('Logical length exceeds native int32 limit')
        self.logical_numel,self.bits=logical_numel,bits
        self.padded_numel=(logical_numel+255)//256*256;self.device=_cuda_device(device)
        n=self.padded_numel
        self.packet=DevicePacket(torch.empty(n*(bits+1)//8,device=self.device,dtype=torch.uint8),
            torch.empty(n//16,device=self.device,dtype=torch.uint8),
            torch.empty(n//256,device=self.device,dtype=torch.bfloat16),logical_numel,n,bits)
        self.decoded_padded=torch.empty(n,device=self.device,dtype=torch.bfloat16)
        self.output=self.decoded_padded[:logical_numel]
        self.status=torch.empty(1,device=self.device,dtype=torch.int32)
        self.epsilon=torch.tensor(1e-23,device=self.device,dtype=torch.bfloat16)
        self.last_quantization_events=0
        self.extension=load_extension()

    def _owned_storage_ids(self):
        values=[v for v in vars(self).values() if isinstance(v,torch.Tensor)]
        values.extend(getattr(self.packet,k) for k in ('payload','scale_codes','top_scales'))
        return {v.untyped_storage().data_ptr() for v in values if v.untyped_storage().nbytes()}

    def _raise_status(self):
        code=int(self.status.item())
        for flag,message in STATUS_MESSAGES:
            if code&flag:raise ValueError(message)
        if code:raise RuntimeError(f'Unknown CUDA codec status: {code}')

    @torch.no_grad()
    def encode_into(self,x,main_thresholds,scale_thresholds,lut,*,checked=False):
        self._encode_metadata(x,main_thresholds,scale_thresholds,lut);self.last_quantization_events=0
        if checked:
            self._check_external_aliases((('input',x),('main thresholds',main_thresholds),('scale thresholds',scale_thresholds),('LUT',lut)))
            self._check_lut(lut)
            _require_device_condition(torch.isfinite(x).all(),'Nonfinite input')
            for threshold in (main_thresholds,scale_thresholds):
                _require_device_condition((torch.isfinite(threshold)&(threshold>=0)&(threshold<=65536)).all(),'Invalid explicit thresholds')
        if not self.logical_numel:return self.packet
        if checked:self.status.zero_()
        self.extension.encode_into(x,main_thresholds,scale_thresholds,lut,self.packet.payload,self.packet.scale_codes,self.packet.top_scales,self.bits,self.status,checked)
        if checked:self._raise_status()
        self.last_quantization_events=1
        return self.packet

    @torch.no_grad()
    def decode_into(self,packet,lut,*,checked=False):
        self._packet_metadata(packet);self._lut_metadata(lut)
        if checked:
            self._check_external_aliases((('LUT',lut),),packet=packet);self._check_lut(lut)
        if not self.logical_numel:return self.output
        if checked:self.status.zero_()
        self.extension.decode_into(packet.payload,packet.scale_codes,packet.top_scales,lut,self.decoded_padded,self.logical_numel,self.bits,self.status,checked)
        if checked:self._raise_status()
        return self.output

    def tensor_inventory(self):
        values=[(k,v,'decoded_output' if k in ('decoded_padded','output') else 'workspace') for k,v in vars(self).items() if isinstance(v,torch.Tensor)]
        values.extend(('packet.'+k,getattr(self.packet,k),'packet') for k in ('payload','scale_codes','top_scales'))
        result=[];seen=set()
        for name,value,category in values:
            s=value.untyped_storage();identity=(s.data_ptr(),s.nbytes())
            if identity in seen:continue
            seen.add(identity);result.append(dict(name=name,category=category,bytes=s.nbytes(),dtype=str(value.dtype),shape=list(value.shape)))
        return result

def _pack_args(count,stored_bits):
    if type(count) is not int or count<0 or type(stored_bits) is not int or stored_bits not in (5,9):
        raise ValueError('This local codec packs only 5/9-bit nonnegative counts')
    if count>2147483647:raise ValueError('Code count exceeds native int32 limit')
    return (count*stored_bits+7)//8

def _upper(stored_bits,max_code):
    upper=(1<<stored_bits)-1 if max_code is None else max_code
    if type(upper) is not int or not 0<=upper<(1<<stored_bits):raise ValueError('Invalid unsigned alphabet limit')
    return upper

def pack_unsigned_cuda(codes,stored_bits,*,checked=True,max_code=None):
    if not isinstance(codes,torch.Tensor) or codes.ndim!=1 or codes.dtype not in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64) or not codes.is_contiguous():
        raise ValueError('Expected a contiguous one-dimensional integer tensor')
    device=_cuda_device(codes.device);byte_count=_pack_args(codes.numel(),stored_bits);upper=_upper(stored_bits,max_code)
    if checked:_require_device_condition(((codes>=0)&(codes<=upper)).all(),'Code outside unsigned alphabet')
    out=torch.empty(byte_count,device=device,dtype=torch.uint8)
    load_extension().pack_into(codes.to(torch.int64),out,stored_bits)
    return out

def unpack_unsigned_cuda(payload,stored_bits,count,*,checked=True,max_code=None):
    if not isinstance(payload,torch.Tensor):raise ValueError('Expected a CUDA payload tensor')
    device=_cuda_device(payload.device);byte_count=_pack_args(count,stored_bits)
    _require_metadata(payload,dtype=torch.uint8,shape=(byte_count,),device=device,name='payload')
    if checked and count*stored_bits%8 and byte_count:
        _require_device_condition((payload[-1].to(torch.int64)>>(count*stored_bits%8))==0,'Nonzero trailing padding bits')
    out=torch.empty(count,device=device,dtype=torch.int64)
    load_extension().unpack_into(payload,out,stored_bits);upper=_upper(stored_bits,max_code)
    if checked:_require_device_condition((out<=upper).all(),'Unused code encountered in packed stream')
    return out

def bf16_add_epsilon(x):
    _require_metadata(x,dtype=torch.bfloat16,shape=(x.numel(),),device=_cuda_device(x.device),name='input')
    out=torch.empty_like(x);load_extension().bf16_add_epsilon_into(x,out);return out

def encode(x,bits,main_thresholds,scale_thresholds,lut,*,checked=False):
    return CudaCodecPlan(x.numel(),bits,x.device).encode_into(x,main_thresholds,scale_thresholds,lut,checked=checked)
def decode(packet,lut,*,checked=False):
    return CudaCodecPlan(packet.logical_numel,packet.bits,packet.payload.device).decode_into(packet,lut,checked=checked)
def roundtrip(x,bits,main_thresholds,scale_thresholds,lut,*,checked=False):
    plan=CudaCodecPlan(x.numel(),bits,x.device)
    return plan.packet,plan.roundtrip_into(x,main_thresholds,scale_thresholds,lut,checked=checked)
