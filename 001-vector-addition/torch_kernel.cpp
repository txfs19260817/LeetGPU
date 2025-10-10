#include <torch/extension.h>

extern "C" void solve(const float *A, const float *B, float *C, int N);

void torch_launch_vector_add(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c, int64_t n) {
  solve((const float *)a.data_ptr(), (const float *)b.data_ptr(), (float *)c.data_ptr(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_vector_add", &torch_launch_vector_add,
        "vector add kernel warpper");
}

TORCH_LIBRARY(vector_add, m) {
  m.def("torch_launch_vector_add", torch_launch_vector_add);
}