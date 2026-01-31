#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>
#include <vector>

// Declarations from custom_all_reduce.cu
int64_t init_custom_ar(
    const std::vector<int64_t>& fake_ipc_ptrs,
    torch::Tensor& rank_data,
    int64_t rank,
    bool full_nvlink);

void all_reduce(
    int64_t fa,
    torch::Tensor& inp,
    torch::Tensor& out,
    int64_t reg_buffer,
    int64_t reg_buffer_sz_bytes);
void tree_all_reduce(
    int64_t fa,
    torch::Tensor& inp,
    torch::Tensor& out,
    int64_t reg_buffer,
    int64_t reg_buffer_sz_bytes);

void dispose(int64_t fa);
int64_t meta_size();
void register_buffer(int64_t fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    int64_t fa);
void register_graph_buffers(
    int64_t fa,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets);

TORCH_LIBRARY_FRAGMENT(tbik_kernel, m) {
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);

  m.def(
      "tree_all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("tree_all_reduce", torch::kCUDA, &tree_all_reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}