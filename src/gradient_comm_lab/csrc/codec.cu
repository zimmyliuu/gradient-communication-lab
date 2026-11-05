#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using at::Tensor;
using BF=__nv_bfloat16;
__device__ __forceinline__ float R(float v){return __bfloat162float(__float2bfloat16_rn(v));}
__device__ __forceinline__ float V(BF v){return __bfloat162float(v);}
__device__ __forceinline__ float add(float a,float b){return R(__fadd_rn(a,b));}
__device__ __forceinline__ float sub(float a,float b){return R(__fsub_rn(a,b));}
__device__ __forceinline__ float mul(float a,float b){return R(__fmul_rn(a,b));}
__device__ __forceinline__ float divb(float a,float b){return R(__fdiv_rn(a,b));}
__device__ __forceinline__ void fail(int* status,bool checked,int bit){if(checked)atomicOr(status,bit);}
// Each CTA owns one complete 256-element block and its whole 160/288-byte payload.
// Group reductions are within fixed 16-lane segments; only shared scales/codes survive barriers.
template<int B> __global__ void encode_kernel(const BF* x,int logical,const BF* thresholds,const BF* scale_thresholds,const BF* lut,unsigned char* payload,unsigned char* scale_codes,BF* top,int* status,bool checked){
  constexpr int W=B+1,BOUND=1<<(B-1),OFFSET=1<<B,BYTES=256*W/8;
  __shared__ float scales[16]; __shared__ unsigned short codes[256]; __shared__ float top_value;
  int t=threadIdx.x,base=blockIdx.x*256,i=base+t,g=t/16;
  float value=i<logical?V(x[i]):0.0f;
  float maximum=fabsf(value);
  for(int d=8;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d,16));
  if((t&15)==0)scales[g]=add(divb(maximum,V(lut[3*BOUND])),R(1e-23f));
  __syncthreads();
  float normalized=divb(value,scales[g]);
  if(!isfinite(normalized))fail(status,checked,1);
  int lo=0,hi=2*OFFSET+1;
  while(lo<hi){int mid=(lo+hi)/2;if(normalized<V(lut[mid]))hi=mid;else lo=mid+1;}
  if(lo<=0 || lo>=2*OFFSET+1)fail(status,checked,2);
  // Guard all loads even in trusted mode; invalid values can never cause out-of-range LUT access.
  int idx=max(1,min(lo,2*OFFSET));
  float left=V(lut[idx-1]),right=V(lut[idx]);
  float probability=divb(sub(normalized,left),sub(right,left));
  if(!isfinite(probability))fail(status,checked,4);
  int chosen=(V(thresholds[i])<mul(probability,65536.0f))?idx:idx-1;
  codes[t]=(unsigned short)(max(-BOUND,min(chosen-OFFSET,BOUND))+BOUND);
  if(t<16){
    float mx=scales[t];
    for(int d=8;d;d>>=1)mx=fmaxf(mx,__shfl_down_sync(0x0000ffff,mx,d,16));
    if(t==0){top_value=add(mx,R(1e-23f));top[blockIdx.x]=__float2bfloat16_rn(top_value);if(!isfinite(top_value)||top_value<=0)fail(status,checked,8);}
  }
  __syncthreads();
  if(t<16){
    float level=mul(divb(scales[t],top_value),255.0f),lower=R(floorf(level));
    float p=mul(sub(level,lower),65536.0f);
    float code=add(lower,V(scale_thresholds[blockIdx.x*16+t])<p?1.0f:0.0f);
    if(!isfinite(code)||code<0||code>255){fail(status,checked,16);code=0;}
    scale_codes[blockIdx.x*16+t]=(unsigned char)code;
  }
  // Unique byte ownership, assembled from at most three neighboring register/shared codes.
  for(int byte=t;byte<BYTES;byte+=256){
    int bit=byte*8,index=bit/W,shift=bit%W;
    unsigned int word=codes[index]>>shift;
    if(index+1<256)word|=(unsigned int)codes[index+1]<<(W-shift);
    if(index+2<256)word|=(unsigned int)codes[index+2]<<(2*W-shift);
    payload[blockIdx.x*BYTES+byte]=(unsigned char)word;
  }
}
template<int B> __global__ void decode_kernel(const unsigned char* payload,const unsigned char* scale_codes,const BF* top,const BF* lut,BF* out,int* status,bool checked){
  constexpr int W=B+1,BOUND=1<<(B-1),OFFSET=1<<B,BYTES=256*W/8;
  int t=threadIdx.x,i=blockIdx.x*256+t,bit=t*W,byte=bit/8,shift=bit%8;
  unsigned int word=payload[blockIdx.x*BYTES+byte];
  if(byte+1<BYTES)word|=(unsigned int)payload[blockIdx.x*BYTES+byte+1]<<8;
  unsigned int code=(word>>shift)&((1<<W)-1);
  if(code>OFFSET)fail(status,checked,32);
  float topv=V(top[blockIdx.x]);
  if(!isfinite(topv)||topv<=0)fail(status,checked,64);
  float scale=divb(mul(R((float)scale_codes[i/16]),topv),255.0f);
  if(!isfinite(scale))fail(status,checked,128);
  // Clamp illegal unsigned codes before deriving a LUT address.
  int index=(int)min(code,(unsigned int)OFFSET)-BOUND+OFFSET;
  float result=mul(V(lut[index]),scale);
  if(!isfinite(result))fail(status,checked,256);
  out[i]=__float2bfloat16_rn(result);
}
__global__ void pack_kernel(const int64_t* codes,int count,unsigned char* payload,int bits,int64_t bytes){
  int64_t byte=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(byte>=bytes)return;
  int64_t bit=byte*8,index=bit/bits;int shift=bit%bits;
  uint64_t word=(uint64_t)codes[index]>>shift;
  if(index+1<count)word|=(uint64_t)codes[index+1]<<(bits-shift);
  if(index+2<count)word|=(uint64_t)codes[index+2]<<(2*bits-shift);
  payload[byte]=(unsigned char)word;
}
__global__ void unpack_kernel(const unsigned char* payload,int64_t* codes,int count,int bits,int64_t bytes){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int64_t bit=int64_t(i)*bits,byte=bit/8;int shift=bit%8;
  unsigned int word=payload[byte];if(byte+1<bytes)word|=(unsigned int)payload[byte+1]<<8;
  codes[i]=(word>>shift)&((1<<bits)-1);
}
__global__ void epsilon_kernel(const BF* x,BF* out,int64_t count){int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)out[i]=__float2bfloat16_rn(add(V(x[i]),R(1e-23f)));}
void encode_cuda(Tensor x,Tensor main,Tensor scale,Tensor lut,Tensor payload,Tensor scales,Tensor top,int bits,Tensor status,bool checked){
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());int blocks=(x.numel()+255)/256;
#define ENCODE(B) encode_kernel<B><<<blocks,256,0,stream>>>((BF*)x.data_ptr(),x.numel(),(BF*)main.data_ptr(),(BF*)scale.data_ptr(),(BF*)lut.data_ptr(),payload.data_ptr<unsigned char>(),scales.data_ptr<unsigned char>(),(BF*)top.data_ptr(),status.data_ptr<int>(),checked)
  if(bits==4){ENCODE(4);}else{ENCODE(8);}C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void decode_cuda(Tensor payload,Tensor scales,Tensor top,Tensor lut,Tensor out,int logical,int bits,Tensor status,bool checked){
  auto stream=c10::cuda::getCurrentCUDAStream(out.get_device());int blocks=out.numel()/256;
#define DECODE(B) decode_kernel<B><<<blocks,256,0,stream>>>(payload.data_ptr<unsigned char>(),scales.data_ptr<unsigned char>(),(BF*)top.data_ptr(),(BF*)lut.data_ptr(),(BF*)out.data_ptr(),status.data_ptr<int>(),checked)
  if(bits==4){DECODE(4);}else{DECODE(8);}C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pack_cuda(Tensor codes,Tensor payload,int bits){auto stream=c10::cuda::getCurrentCUDAStream(codes.get_device());pack_kernel<<<(payload.numel()+255)/256,256,0,stream>>>(codes.data_ptr<int64_t>(),codes.numel(),payload.data_ptr<unsigned char>(),bits,payload.numel());C10_CUDA_KERNEL_LAUNCH_CHECK();}
void unpack_cuda(Tensor payload,Tensor codes,int bits){auto stream=c10::cuda::getCurrentCUDAStream(codes.get_device());unpack_kernel<<<(codes.numel()+255)/256,256,0,stream>>>(payload.data_ptr<unsigned char>(),codes.data_ptr<int64_t>(),codes.numel(),bits,payload.numel());C10_CUDA_KERNEL_LAUNCH_CHECK();}
void epsilon_cuda(Tensor x,Tensor out){auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());epsilon_kernel<<<(x.numel()+255)/256,256,0,stream>>>((BF*)x.data_ptr(),(BF*)out.data_ptr(),x.numel());C10_CUDA_KERNEL_LAUNCH_CHECK();}
