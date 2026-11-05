#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
__global__ void increment_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}
torch::Tensor cuda_increment(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
              "G0 probe requires contiguous CUDA FP32 input");
  auto y = torch::empty_like(x);
  increment_kernel<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(x.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
