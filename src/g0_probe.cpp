#include <torch/extension.h>
torch::Tensor cuda_increment(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("increment", &cuda_increment); }
