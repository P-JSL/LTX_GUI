// Fused CUDA kernels for dense BCTHW tensors and per-frame/spatial masks.
#include <torch/extension.h>
#include <climits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// The output buffer is disjoint from every input. Each thread processes one
// element, broadcasts the mask across channels, and writes exactly once.
template <typename scalar_t, bool BLEND>
__global__ void masked_kernel(const scalar_t* a, const scalar_t* b, const scalar_t* known,
                              const float* mask, scalar_t* output,
                              int64_t total, int64_t channels, int64_t pixels, float step) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t batch = idx / (channels * pixels);
    int64_t pos = batch * pixels + idx % pixels;
    float av = static_cast<float>(a[idx]);
    float bv = static_cast<float>(b[idx]);
    float mv = mask[pos];
    float value;
    if constexpr (BLEND) {
        value = av + (bv - av) * mv;
    } else {
        value = (av - bv * step) * mv + static_cast<float>(known[idx]);
    }
    output[idx] = static_cast<scalar_t>(value);
}

template <bool BLEND>
at::Tensor dispatch_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& known,
                         const at::Tensor& mask, double step) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto out = at::empty_like(a);
    const int64_t total = a.numel();
    if (total == 0) return out;
    const int64_t pixels = a.size(2) * a.size(3) * a.size(4);
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= INT_MAX, "Too many CUDA blocks");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "ltx_gui_masked", [&] {
        masked_kernel<scalar_t, BLEND><<<static_cast<int>(blocks), threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            known.data_ptr<scalar_t>(), mask.data_ptr<float>(), out.data_ptr<scalar_t>(),
            total, a.size(1), pixels, static_cast<float>(step));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor update_cuda(at::Tensor z, at::Tensor velocity, at::Tensor known, at::Tensor mask, double step) {
    return dispatch_cuda<false>(z, velocity, known, mask, step);
}

at::Tensor blend_cuda(at::Tensor original, at::Tensor restored, at::Tensor mask) {
    return dispatch_cuda<true>(original, restored, restored, mask, 0.0);
}
