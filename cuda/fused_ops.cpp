// Independent CUDA extension entry points; runtime build only, no third-party source.
#include <torch/extension.h>
#include <stdexcept>

void check_inputs(const at::Tensor& z, const at::Tensor& b, const at::Tensor& k, const at::Tensor& mask) {
    TORCH_CHECK(z.is_cuda() && b.is_cuda() && k.is_cuda() && mask.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(z.is_contiguous() && b.is_contiguous() && k.is_contiguous() && mask.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(z.dim() == 5 && z.sizes() == b.sizes() && z.sizes() == k.sizes(), "tensor shapes mismatch");
    TORCH_CHECK(mask.dim() == 5 && mask.size(0) == z.size(0) && mask.size(1) == 1 &&
        mask.size(2) == z.size(2) && mask.size(3) == z.size(3) && mask.size(4) == z.size(4), "mask shape mismatch");
    TORCH_CHECK(z.device() == b.device() && z.device() == k.device() && z.device() == mask.device(), "device mismatch");
    TORCH_CHECK(z.scalar_type() == b.scalar_type() && z.scalar_type() == k.scalar_type() &&
        mask.scalar_type() == at::kFloat, "dtype mismatch");
    TORCH_CHECK(z.scalar_type() == at::kFloat || z.scalar_type() == at::kHalf || z.scalar_type() == at::kBFloat16,
        "unsupported CUDA dtype");
}

at::Tensor update_cuda(at::Tensor z, at::Tensor velocity, at::Tensor known, at::Tensor mask, double step);
at::Tensor blend_cuda(at::Tensor original, at::Tensor restored, at::Tensor mask);

at::Tensor masked_update(at::Tensor z, at::Tensor velocity, at::Tensor known, at::Tensor mask, double step) {
    check_inputs(z, velocity, known, mask);
    return update_cuda(z, velocity, known, mask, step);
}

at::Tensor masked_composite(at::Tensor original, at::Tensor restored, at::Tensor mask) {
    check_inputs(original, restored, restored, mask);
    return blend_cuda(original, restored, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_update", &masked_update, "Fused masked Euler diffusion update (CUDA)");
    m.def("masked_composite", &masked_composite, "Fused pixel compositing (CUDA)");
}
