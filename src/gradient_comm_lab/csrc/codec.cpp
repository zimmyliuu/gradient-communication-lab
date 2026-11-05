#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <vector>

using torch::Tensor;
void encode_cuda(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int,Tensor,bool);
void decode_cuda(Tensor,Tensor,Tensor,Tensor,Tensor,int,int,Tensor,bool);
void pack_cuda(Tensor,Tensor,int);
void unpack_cuda(Tensor,Tensor,int);
void epsilon_cuda(Tensor,Tensor);

static void tensor(const Tensor& t, at::ScalarType dtype, int64_t count, const Tensor& anchor, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype && t.numel()==count && t.device()==anchor.device(), "ABI_METADATA: ",name);
}
static void no_alias(const std::vector<Tensor>& reads, const std::vector<Tensor>& writes) {
  for(size_t i=0;i<writes.size();++i) {
    for(const auto& r: reads) TORCH_CHECK(!writes[i].numel() || !r.numel() || !writes[i].is_alias_of(r),"ABI_ALIAS: output overlaps input storage");
    for(size_t j=0;j<i;++j) TORCH_CHECK(!writes[i].numel() || !writes[j].numel() || !writes[i].is_alias_of(writes[j]),"ABI_ALIAS: outputs share storage");
  }
}
static void bits_check(int bits) { TORCH_CHECK(bits==4 || bits==8,"ABI_BITS: expected nominal b4/b8"); }
// Keep allocator-owned backing storage alive until the caller's current stream
// completes, including temporaries created on a different producer stream.
// Dependency ordering between streams remains explicit at the caller boundary.
static void record_use(const std::vector<Tensor>& tensors) {
  for(const auto& t:tensors) if(t.numel())
    c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),c10::cuda::getCurrentCUDAStream(t.get_device()));
}
void encode(Tensor x,Tensor main,Tensor scale,Tensor lut,Tensor payload,Tensor scales,Tensor top,int bits,Tensor status,bool checked) {
  bits_check(bits); TORCH_CHECK(x.dim()==1 && x.numel()<=INT32_MAX-255,"ABI_LENGTH: input must be one-dimensional and fit int32");
  int64_t n=(x.numel()+255)/256*256;
  tensor(x,at::kBFloat16,x.numel(),x,"input");tensor(main,at::kBFloat16,n,x,"main thresholds");tensor(scale,at::kBFloat16,n/16,x,"scale thresholds");
  tensor(lut,at::kBFloat16,2*(1<<bits)+1,x,"LUT");tensor(payload,at::kByte,n*(bits+1)/8,x,"payload");tensor(scales,at::kByte,n/16,x,"scale codes");tensor(top,at::kBFloat16,n/256,x,"top scales");tensor(status,at::kInt,1,x,"status");
  no_alias({x,main,scale,lut},{payload,scales,top,status});
  c10::cuda::CUDAGuard guard(x.device()); if(n){record_use({x,main,scale,lut,payload,scales,top,status});encode_cuda(x,main,scale,lut,payload,scales,top,bits,status,checked);}
}
void decode(Tensor payload,Tensor scales,Tensor top,Tensor lut,Tensor out,int logical,int bits,Tensor status,bool checked) {
  bits_check(bits);TORCH_CHECK(logical>=0 && logical<=INT32_MAX-255,"ABI_LENGTH: invalid logical length");int64_t n=(int64_t(logical)+255)/256*256;
  tensor(out,at::kBFloat16,n,out,"output");tensor(payload,at::kByte,n*(bits+1)/8,out,"payload");tensor(scales,at::kByte,n/16,out,"scale codes");tensor(top,at::kBFloat16,n/256,out,"top scales");tensor(lut,at::kBFloat16,2*(1<<bits)+1,out,"LUT");tensor(status,at::kInt,1,out,"status");
  no_alias({payload,scales,top,lut},{out,status});
  c10::cuda::CUDAGuard guard(out.device());if(n){record_use({payload,scales,top,lut,out,status});decode_cuda(payload,scales,top,lut,out,logical,bits,status,checked);}
}
void pack(Tensor codes,Tensor payload,int bits) {
  TORCH_CHECK(bits==5 || bits==9,"ABI_BITS: expected stored 5/9 bits");TORCH_CHECK(codes.dim()==1 && codes.numel()<=INT32_MAX,"ABI_LENGTH: invalid code count");
  tensor(codes,at::kLong,codes.numel(),codes,"codes");tensor(payload,at::kByte,(codes.numel()*bits+7)/8,codes,"payload");no_alias({codes},{payload});
  c10::cuda::CUDAGuard guard(codes.device());if(codes.numel()){record_use({codes,payload});pack_cuda(codes,payload,bits);}
}
void unpack(Tensor payload,Tensor codes,int bits) {
  TORCH_CHECK(bits==5 || bits==9,"ABI_BITS: expected stored 5/9 bits");TORCH_CHECK(codes.dim()==1 && codes.numel()<=INT32_MAX,"ABI_LENGTH: invalid code count");
  tensor(codes,at::kLong,codes.numel(),codes,"codes");tensor(payload,at::kByte,(codes.numel()*bits+7)/8,codes,"payload");no_alias({payload},{codes});
  c10::cuda::CUDAGuard guard(codes.device());if(codes.numel()){record_use({payload,codes});unpack_cuda(payload,codes,bits);}
}
void epsilon(Tensor x,Tensor out) {
  tensor(x,at::kBFloat16,x.numel(),x,"input");tensor(out,at::kBFloat16,x.numel(),x,"output");no_alias({x},{out});c10::cuda::CUDAGuard guard(x.device());if(x.numel()){record_use({x,out});epsilon_cuda(x,out);}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("encode_into",&encode);m.def("decode_into",&decode);m.def("pack_into",&pack);m.def("unpack_into",&unpack);m.def("bf16_add_epsilon_into",&epsilon);
}
