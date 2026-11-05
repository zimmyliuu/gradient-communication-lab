"""Fixed local b4/b8 codec, pure PyTorch CUDA eager, explicit thresholds.

Trusted methods read tensor metadata only on the host. All intermediate arrays
are preallocated, and all arithmetic, packing and unpacking stays on device.
Checked methods additionally synchronize device validation predicates at the
API boundary and between unsafe stages; their cost is a separate API result.
This implements neither the DynamiQ controller nor any collective or RNG pool.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

METHOD_ID = 'torch_cuda_repaired'


def _cuda_device(device):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('This implementation requires CUDA tensors')
    return torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)


def _require_metadata(x, *, dtype, shape, device, name):
    if not isinstance(x, torch.Tensor) or x.dtype != dtype or tuple(x.shape) != tuple(shape):
        raise ValueError(f'{name}: unexpected tensor dtype or shape')
    if x.device != device or not x.is_contiguous():
        raise ValueError(f'{name}: expected contiguous tensor on {device}')


def _require_device_condition(condition, message):
    """Explicit checked-API synchronization; NEVER called by trusted methods."""
    if not bool(condition):
        raise ValueError(message)


class _PackWorkspace:
    def __init__(self, count, bits, device):
        if type(count) is not int or count < 0 or type(bits) is not int or bits not in (5, 9):
            raise ValueError('This local codec packs only 5/9-bit nonnegative counts')
        self.count, self.bits, self.device = count, bits, device
        self.byte_count = (count*bits+7)//8
        byte_bit = torch.arange(self.byte_count, device=device, dtype=torch.int64)*8
        self.index0 = torch.div(byte_bit, bits, rounding_mode='floor')
        self.index1 = self.index0+1
        self.index2 = self.index0+2
        self.shift0 = torch.remainder(byte_bit, bits)
        self.shift1 = bits-self.shift0
        self.shift2 = 2*bits-self.shift0
        self.codes_pad = torch.zeros(count+2, device=device, dtype=torch.int64)
        self.word0 = torch.empty(self.byte_count, device=device, dtype=torch.int64)
        self.word1 = torch.empty_like(self.word0)
        self.word2 = torch.empty_like(self.word0)
        element_bit = torch.arange(count, device=device, dtype=torch.int64)*bits
        self.byte_index0 = torch.div(element_bit, 8, rounding_mode='floor')
        self.byte_index1 = self.byte_index0+1
        self.element_shift = torch.remainder(element_bit, 8)
        self.bytes_pad = torch.zeros(self.byte_count+1, device=device, dtype=torch.int64)
        self.element_word = torch.empty(count, device=device, dtype=torch.int64)

    def pack_into(self, codes, payload):
        if not self.count:
            return payload
        self.codes_pad[:self.count].copy_(codes)
        torch.index_select(self.codes_pad, 0, self.index0, out=self.word0)
        torch.index_select(self.codes_pad, 0, self.index1, out=self.word1)
        torch.index_select(self.codes_pad, 0, self.index2, out=self.word2)
        torch.bitwise_right_shift(self.word0, self.shift0, out=self.word0)
        torch.bitwise_left_shift(self.word1, self.shift1, out=self.word1)
        torch.bitwise_left_shift(self.word2, self.shift2, out=self.word2)
        self.word0.bitwise_or_(self.word1).bitwise_or_(self.word2).bitwise_and_(255)
        payload.copy_(self.word0)
        return payload

    def unpack_into(self, payload, codes):
        if not self.count:
            return codes
        self.bytes_pad[:self.byte_count].copy_(payload)
        torch.index_select(self.bytes_pad, 0, self.byte_index0, out=codes)
        torch.index_select(self.bytes_pad, 0, self.byte_index1, out=self.element_word)
        self.element_word.bitwise_left_shift_(8)
        codes.bitwise_or_(self.element_word)
        torch.bitwise_right_shift(codes, self.element_shift, out=codes)
        codes.bitwise_and_((1 << self.bits)-1)
        return codes


def pack_unsigned_cuda(codes, stored_bits, *, checked=True, max_code=None):
    """Allocating device pack helper for code-alphabet tests, not main timing."""
    if not isinstance(codes, torch.Tensor) or codes.ndim != 1 or codes.dtype not in (
            torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64) or not codes.is_contiguous():
        raise ValueError('Expected a contiguous one-dimensional integer tensor')
    device = _cuda_device(codes.device)
    workspace = _PackWorkspace(codes.numel(), stored_bits, device)
    upper = (1 << stored_bits)-1 if max_code is None else max_code
    if type(upper) is not int or not 0 <= upper < (1 << stored_bits):
        raise ValueError('Invalid unsigned alphabet limit')
    if checked:
        _require_device_condition(((codes >= 0) & (codes <= upper)).all(), 'Code outside unsigned alphabet')
    out = torch.empty(workspace.byte_count, device=device, dtype=torch.uint8)
    return workspace.pack_into(codes, out)


def unpack_unsigned_cuda(payload, stored_bits, count, *, checked=True, max_code=None):
    """Allocating device unpack helper, including canonical trailing-bit checks."""
    if not isinstance(payload, torch.Tensor):
        raise ValueError('Expected a CUDA payload tensor')
    device = _cuda_device(payload.device)
    workspace = _PackWorkspace(count, stored_bits, device)
    _require_metadata(payload, dtype=torch.uint8, shape=(workspace.byte_count,), device=device, name='payload')
    if checked and count*stored_bits % 8 and workspace.byte_count:
        _require_device_condition((payload[-1].to(torch.int64) >> (count*stored_bits % 8)) == 0,
                                  'Nonzero trailing padding bits')
    out = torch.empty(count, device=device, dtype=torch.int64)
    workspace.unpack_into(payload, out)
    upper = (1 << stored_bits)-1 if max_code is None else max_code
    if type(upper) is not int or not 0 <= upper < (1 << stored_bits):
        raise ValueError('Invalid unsigned alphabet limit')
    if checked:
        _require_device_condition((out <= upper).all(), 'Unused code encountered in packed stream')
    return out


@dataclass
class DevicePacket:
    payload: torch.Tensor
    scale_codes: torch.Tensor
    top_scales: torch.Tensor
    logical_numel: int
    padded_numel: int
    bits: int

    @property
    def wire_bytes(self):
        return self.payload.numel()+self.scale_codes.numel()+2*self.top_scales.numel()

    def serialized(self):
        """Compact device uint8 serialization; allocation is outside into timing."""
        return torch.cat((self.payload, self.scale_codes, self.top_scales.view(torch.uint8)))


class TorchCodecPlan:
    """Owned preallocated workspace and output buffers for a fixed N and b.

    The tensor inventory includes persistent index arrays and scalar constants.
    Inputs, thresholds and the supplied LUT are external and not counted here.
    Trusted inputs must have passed CPU roundtrip and checked-API validation.
    Users must not alias external input tensors with this plan's owned storage.
    """
    def __init__(self, logical_numel, bits, device='cuda'):
        if type(logical_numel) is not int or logical_numel < 0 or type(bits) is not int or bits not in (4,8):
            raise ValueError('Expected nonnegative logical length and nominal b4/b8')
        self.logical_numel, self.bits = logical_numel, bits
        self.padded_numel = (logical_numel+255)//256*256
        self.device = _cuda_device(device)
        n, groups, top_groups = self.padded_numel, self.padded_numel//16, self.padded_numel//256
        self.pack_workspace = _PackWorkspace(n,bits+1,self.device)
        self.packet = DevicePacket(torch.empty((n*(bits+1)+7)//8,device=self.device,dtype=torch.uint8),
            torch.empty(groups,device=self.device,dtype=torch.uint8),
            torch.empty(top_groups,device=self.device,dtype=torch.bfloat16),logical_numel,n,bits)
        self.decoded_padded = torch.empty(n,device=self.device,dtype=torch.bfloat16)
        self.output = self.decoded_padded[:logical_numel]
        self.work = torch.empty(n,device=self.device,dtype=torch.bfloat16)
        self.abs_work = torch.empty_like(self.work)
        self.scales = torch.empty(groups,device=self.device,dtype=torch.bfloat16)
        self.left = torch.empty_like(self.work)
        self.right = torch.empty_like(self.work)
        self.probability = torch.empty_like(self.work)
        self.probability_scaled = torch.empty_like(self.work)
        self.indices = torch.empty(n,device=self.device,dtype=torch.int64)
        self.left_indices = torch.empty_like(self.indices)
        self.signed_codes = torch.empty_like(self.indices)
        self.unsigned_codes = torch.empty_like(self.indices)
        self.comparison = torch.empty(n,device=self.device,dtype=torch.bool)
        self.levels = torch.empty(groups,device=self.device,dtype=torch.bfloat16)
        self.lower = torch.empty_like(self.levels)
        self.scale_probability = torch.empty_like(self.levels)
        self.scale_comparison = torch.empty(groups,device=self.device,dtype=torch.bool)
        self.decoded_scales = torch.empty_like(self.levels)
        self.epsilon = torch.tensor(1e-23,dtype=torch.bfloat16,device=self.device)
        self.random_multiplier = torch.tensor(65535,dtype=torch.bfloat16,device=self.device)
        self.scale_multiplier = torch.tensor(255,dtype=torch.bfloat16,device=self.device)
        self.last_quantization_events = 0

    def _lut_metadata(self, lut):
        _require_metadata(lut,dtype=torch.bfloat16,shape=(2*(1 << self.bits)+1,),device=self.device,name='LUT')

    def _check_lut(self,lut):
        _require_device_condition(torch.isfinite(lut).all() & (lut[1:] > lut[:-1]).all(),
                                  'LUT must be finite and strictly ascending')

    def _owned_storage_ids(self):
        # Storage identity is host metadata: no device values are inspected.
        tensors=[value for value in vars(self).values() if isinstance(value,torch.Tensor)]
        tensors.extend(value for value in vars(self.pack_workspace).values() if isinstance(value,torch.Tensor))
        tensors.extend(getattr(self.packet,name) for name in ('payload','scale_codes','top_scales'))
        return {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.untyped_storage().nbytes()}

    def _check_external_aliases(self,tensors,*,packet=None):
        owned=self._owned_storage_ids()
        for name,tensor in tensors:
            if tensor.untyped_storage().nbytes() and tensor.untyped_storage().data_ptr() in owned:
                raise ValueError(f'{name}: aliases plan-owned storage')
        if packet is not None:
            for name in ('payload','scale_codes','top_scales'):
                tensor=getattr(packet,name)
                storage=tensor.untyped_storage()
                own_packet_storage=getattr(self.packet,name).untyped_storage().data_ptr()
                if storage.nbytes() and storage.data_ptr() in owned and storage.data_ptr()!=own_packet_storage:
                    raise ValueError(f'packet {name}: aliases another plan-owned buffer')

    def _encode_metadata(self,x,main,scale,lut):
        _require_metadata(x,dtype=torch.bfloat16,shape=(self.logical_numel,),device=self.device,name='input')
        _require_metadata(main,dtype=torch.bfloat16,shape=(self.padded_numel//16,16),device=self.device,name='main thresholds')
        _require_metadata(scale,dtype=torch.bfloat16,shape=(self.padded_numel//256,16),device=self.device,name='scale thresholds')
        self._lut_metadata(lut)

    def _packet_metadata(self,packet):
        if not isinstance(packet,DevicePacket):
            raise ValueError('Expected a DevicePacket')
        if (packet.bits != self.bits or packet.padded_numel != self.padded_numel or
                packet.logical_numel != self.logical_numel):
            raise ValueError('Packet identity differs from plan')
        _require_metadata(packet.payload,dtype=torch.uint8,shape=((self.padded_numel*(self.bits+1)+7)//8,),device=self.device,name='payload')
        _require_metadata(packet.scale_codes,dtype=torch.uint8,shape=(self.padded_numel//16,),device=self.device,name='scale codes')
        _require_metadata(packet.top_scales,dtype=torch.bfloat16,shape=(self.padded_numel//256,),device=self.device,name='top scales')

    @torch.no_grad()
    def encode_into(self,x,main_thresholds,scale_thresholds,lut,*,checked=False):
        self._encode_metadata(x,main_thresholds,scale_thresholds,lut)
        self.last_quantization_events = 0
        if checked:
            self._check_external_aliases((('input',x),('main thresholds',main_thresholds),
                ('scale thresholds',scale_thresholds),('LUT',lut)))
            self._check_lut(lut)
            _require_device_condition(torch.isfinite(x).all(),'Nonfinite input')
            for threshold in (main_thresholds,scale_thresholds):
                _require_device_condition((torch.isfinite(threshold) & (threshold >= 0) & (threshold <= 65536)).all(),
                                          'Invalid explicit thresholds')
        if not self.logical_numel:
            return self.packet
        self.work[:self.logical_numel].copy_(x)
        if self.logical_numel != self.padded_numel:
            self.work[self.logical_numel:].zero_()
        torch.abs(self.work,out=self.abs_work)
        torch.amax(self.abs_work.view(-1,16),dim=1,out=self.scales)
        self.scales.div_(lut[(1 << self.bits)+(1 << (self.bits-1))]).add_(self.epsilon)
        self.work.view(-1,16).div_(self.scales.view(-1,1))
        if checked:
            _require_device_condition(torch.isfinite(self.work).all(),'Nonfinite normalized input')
        torch.searchsorted(lut,self.work,right=True,out=self.indices)
        if checked:
            _require_device_condition(((self.indices > 0) & (self.indices < lut.numel())).all(),
                                      'Search index out of range')
        torch.sub(self.indices,1,out=self.left_indices)
        torch.take(lut,self.left_indices,out=self.left)
        torch.take(lut,self.indices,out=self.right)
        torch.sub(self.work,self.left,out=self.probability)
        self.right.sub_(self.left)
        self.probability.div_(self.right)
        if checked:
            _require_device_condition(torch.isfinite(self.probability).all(),'Nonfinite rounding probability')
        torch.mul(self.probability,self.random_multiplier,out=self.probability_scaled)
        torch.lt(main_thresholds.view(-1),self.probability_scaled,out=self.comparison)
        torch.where(self.comparison,self.indices,self.left_indices,out=self.signed_codes)
        self.signed_codes.sub_(1 << self.bits).clamp_(-(1 << (self.bits-1)),1 << (self.bits-1))
        torch.add(self.signed_codes,1 << (self.bits-1),out=self.unsigned_codes)
        torch.amax(self.scales.view(-1,16),dim=1,out=self.packet.top_scales)
        self.packet.top_scales.add_(self.epsilon)
        torch.div(self.scales.view(-1,16),self.packet.top_scales.view(-1,1),out=self.levels.view(-1,16))
        self.levels.mul_(self.scale_multiplier)
        torch.floor(self.levels,out=self.lower)
        torch.sub(self.levels,self.lower,out=self.scale_probability)
        self.scale_probability.mul_(self.random_multiplier)
        torch.lt(scale_thresholds.view(-1),self.scale_probability,out=self.scale_comparison)
        self.lower.add_(self.scale_comparison)
        if checked:
            _require_device_condition((torch.isfinite(self.packet.top_scales) & (self.packet.top_scales > 0)).all(),
                                      'Invalid top scale')
            _require_device_condition((torch.isfinite(self.lower) & (self.lower >= 0) & (self.lower <= 255)).all(),
                                      'Invalid scale code')
        self.packet.scale_codes.copy_(self.lower)
        self.pack_workspace.pack_into(self.unsigned_codes,self.packet.payload)
        self.last_quantization_events = 1
        return self.packet

    @torch.no_grad()
    def decode_into(self,packet,lut,*,checked=False):
        self._packet_metadata(packet);self._lut_metadata(lut)
        if checked:
            self._check_external_aliases((('LUT',lut),),packet=packet)
            self._check_lut(lut)
        if not self.logical_numel:return self.output
        if checked:
            _require_device_condition((torch.isfinite(packet.top_scales) & (packet.top_scales > 0)).all(),
                                      'Nonfinite or nonpositive top scale')
        self.pack_workspace.unpack_into(packet.payload,self.unsigned_codes)
        if checked:
            _require_device_condition((self.unsigned_codes <= (1 << self.bits)).all(),'Unused packed code')
        torch.sub(self.unsigned_codes,1 << (self.bits-1),out=self.signed_codes)
        self.decoded_scales.copy_(packet.scale_codes)
        self.decoded_scales.view(-1,16).mul_(packet.top_scales.view(-1,1))
        self.decoded_scales.div_(self.scale_multiplier)
        if checked:
            _require_device_condition(torch.isfinite(self.decoded_scales).all(),'Nonfinite decoded scales')
        torch.add(self.signed_codes,1 << self.bits,out=self.indices)
        torch.take(lut,self.indices,out=self.decoded_padded)
        self.decoded_padded.view(-1,16).mul_(self.decoded_scales.view(-1,1))
        if checked:
            _require_device_condition(torch.isfinite(self.decoded_padded).all(),'Nonfinite decoded output')
        return self.output

    @torch.no_grad()
    def roundtrip_into(self,x,main_thresholds,scale_thresholds,lut,*,checked=False):
        packet=self.encode_into(x,main_thresholds,scale_thresholds,lut,checked=checked)
        return self.decode_into(packet,lut,checked=checked)

    def tensor_inventory(self):
        """Unique owned storages, excluding external input/threshold/LUT tensors."""
        tensors=[]
        for name,value in vars(self).items():
            if isinstance(value,torch.Tensor):
                tensors.append((name,value,'decoded_output' if name in ('decoded_padded','output') else 'workspace'))
        for name,value in vars(self.pack_workspace).items():
            if isinstance(value,torch.Tensor):tensors.append(('packing.'+name,value,'workspace'))
        for name in ('payload','scale_codes','top_scales'):
            tensors.append(('packet.'+name,getattr(self.packet,name),'packet'))
        result=[];seen=set()
        for name,value,category in tensors:
            storage=value.untyped_storage();identity=(storage.data_ptr(),storage.nbytes())
            if identity in seen:continue
            seen.add(identity)
            result.append({'name':name,'category':category,'bytes':storage.nbytes(),
                'dtype':str(value.dtype),'shape':list(value.shape)})
        return result

    @property
    def workspace_nbytes(self):
        return sum(row['bytes'] for row in self.tensor_inventory() if row['category']=='workspace')


def encode(x,bits,main_thresholds,scale_thresholds,lut,*,checked=False):
    return TorchCodecPlan(x.numel(),bits,x.device).encode_into(x,main_thresholds,scale_thresholds,lut,checked=checked)


def decode(packet,lut,*,checked=False):
    return TorchCodecPlan(packet.logical_numel,packet.bits,packet.payload.device).decode_into(packet,lut,checked=checked)


def roundtrip(x,bits,main_thresholds,scale_thresholds,lut,*,checked=False):
    plan=TorchCodecPlan(x.numel(),bits,x.device)
    return plan.packet,plan.roundtrip_into(x,main_thresholds,scale_thresholds,lut,checked=checked)
