"""
Sample Sort with DeviceSegmentedRadixSort: partition into 128 value-range buckets
using quantile-based splitters, scatter, then sort all buckets in one fused
DeviceSegmentedRadixSort call instead of 128 individual SortKeys.

Algorithm:
  1. Sample 1024 elements, sort on host, pick 127 quantile splitters
  2. Build global histogram (block-local shared mem, reduce to global via atomicAdd)
  3. Host-side exclusive scan on histogram -> per-bucket segment offsets
  4. Scatter: grid-stride atomicAdd on per-bucket counters -> scatter_buf
  5. Single DeviceSegmentedRadixSort::SortKeys sorts all segments in one fused op
  6. No per-bucket host round-trips or individual kernel launches
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda/std/functional>
#include <cstdint>
#include <cstring>

constexpr int NUM_BUCKETS = 128;
constexpr int NUM_SPLITTERS = NUM_BUCKETS - 1;
constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_BLOCK = 1024;

// Persistent CUB temp storage sized for full N (SegmentedRadixSort)
static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    size_t bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes,
        static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(100'000'000),
        128,  // max segments
        static_cast<const int32_t*>(nullptr), static_cast<const int32_t*>(nullptr),
        0, 32);
    persistent_temp_bytes = (bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

__device__ __forceinline__ int find_bucket(
    float val, const float* __restrict__ splitters)
{
    int bucket = 0;
    #pragma unroll
    for (int s = 0; s < NUM_SPLITTERS; s++) {
        bucket += (val >= splitters[s]) ? 1 : 0;
    }
    return bucket;
}

__global__ void histogram_kernel(
    const int32_t* __restrict__ keys,
    const float* __restrict__ splitters,
    int* __restrict__ global_histogram,
    int num_elements)
{
    __shared__ int local_h[NUM_BUCKETS];
    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) local_h[i] = 0;
    __syncthreads();

    int start = blockIdx.x * ITEMS_PER_BLOCK;
    int end = min(start + ITEMS_PER_BLOCK, num_elements);

    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = __int_as_float(keys[i]);
        int b = find_bucket(v, splitters);
        atomicAdd(&local_h[b], 1);
    }
    __syncthreads();

    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        if (local_h[i] > 0) atomicAdd(&global_histogram[i], local_h[i]);
    }
}

__global__ void scatter_kernel(
    const int32_t* __restrict__ keys,
    const float* __restrict__ splitters,
    int32_t* __restrict__ scatter_buf,
    int* __restrict__ counters,
    int num_elements)
{
    int tid = threadIdx.x;
    int start = blockIdx.x * ITEMS_PER_BLOCK;
    int end = min(start + ITEMS_PER_BLOCK, num_elements);

    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = __int_as_float(keys[i]);
        int b = find_bucket(v, splitters);
        int pos = atomicAdd(&counters[b], 1);
        scatter_buf[pos] = keys[i];
    }
}

torch::Tensor sort_cuda(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor splitters_tensor,
    torch::Tensor histogram,
    torch::Tensor bucket_offsets,
    torch::Tensor scatter_buf,
    torch::Tensor seg_begin,
    torch::Tensor seg_end)
{
    int64_t N = input.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in  = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t*       key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());
    const float*   d_splitters = splitters_tensor.const_data_ptr<float>();
    int*           d_histogram = histogram.data_ptr<int>();
    int*           d_offsets   = bucket_offsets.data_ptr<int>();
    int32_t*       d_scatter   = scatter_buf.data_ptr<int32_t>();

    if (N <= static_cast<int64_t>(NUM_BUCKETS * 32)) {
        // Fallback to simple SortKeys for tiny N
        size_t bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, bytes,
            static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<int64_t>(N), 0, 32);
        bytes = (bytes * 11 + 9) / 10;
        auto temp = torch::empty({static_cast<int64_t>(bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        cub::DeviceRadixSort::SortKeys(temp.data_ptr(), bytes,
            key_in, key_out, N, 0, 32, stream);
        cudaStreamSynchronize(stream);
        return output;
    }

    int num_blocks = min(
        (static_cast<int>(N) + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK,
        65535);

    // ====== Step 1: Histogram ======
    cudaMemsetAsync(d_histogram, 0, NUM_BUCKETS * sizeof(int), stream);
    histogram_kernel<<<num_blocks, BLOCK_SIZE, NUM_BUCKETS * sizeof(int), stream>>>(
        key_in, d_splitters, d_histogram, static_cast<int>(N));

    // ====== Step 2: Host-side exclusive scan ======
    int h_hist[NUM_BUCKETS];
    cudaMemcpy(h_hist, d_histogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);

    int h_offsets[NUM_BUCKETS + 1];
    h_offsets[0] = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        h_offsets[b + 1] = h_offsets[b] + h_hist[b];
    }

    // Count non-empty buckets for segment array sizing
    int num_segs = 0;
    int h_begin[NUM_BUCKETS], h_end[NUM_BUCKETS];
    for (int b = 0; b < NUM_BUCKETS; b++) {
        if (h_hist[b] > 0) {
            h_begin[num_segs] = h_offsets[b];
            h_end[num_segs]   = h_offsets[b + 1];
            num_segs++;
        }
    }

    // ====== Step 3: Scatter ======
    // Upload offsets as initial counter values
    cudaMemcpyAsync(d_offsets, h_offsets, (NUM_BUCKETS + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        key_in, d_splitters, d_scatter, d_offsets,
        static_cast<int>(N));

    // ====== Step 4: Segmented sort all buckets in one call ======
    // Upload segment boundaries
    cudaMemcpyAsync(seg_begin.data_ptr(), h_begin, num_segs * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(seg_end.data_ptr(), h_end, num_segs * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Wait for scatter + segment uploads
    cudaStreamSynchronize(stream);

    const int32_t* d_begin = seg_begin.data_ptr<int32_t>();
    const int32_t* d_end   = seg_end.data_ptr<int32_t>();

    size_t temp_bytes = persistent_temp_bytes;
    cudaError_t err = cub::DeviceSegmentedRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        d_scatter, key_out,
        static_cast<int>(N),
        num_segs,
        d_begin, d_end,
        0, 32, stream);

    if (err != cudaSuccess) {
        printf("CUB SegmentedRadixSort failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamSynchronize(stream);
    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output,
                        torch::Tensor splitters,
                        torch::Tensor histogram,
                        torch::Tensor bucket_offsets,
                        torch::Tensor scatter_buf,
                        torch::Tensor seg_begin,
                        torch::Tensor seg_end);
"""

sort_module = load_inline(
    name='sort_cuda_sample_sort_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sample Sort: partition via quantile splitters into 128 buckets,
    scatter, then single DeviceSegmentedRadixSort sorts all buckets at once.
    """
    input_tensor, output_tensor = data
    input_tensor = input_tensor.contiguous()
    N = int(input_tensor.numel())

    # Sample + splitters on host
    sample_size = min(1024, max(128, N // 4))
    sample_idx = torch.randint(0, N, (sample_size,))
    sample_vals = input_tensor.flatten()[sample_idx].cpu().numpy()
    sample_vals.sort()

    splitters = [
        float(sample_vals[(i + 1) * sample_size // 128 - 1])
        for i in range(127)
    ]
    splitters_tensor = torch.tensor(splitters, dtype=torch.float32, device='cuda')

    # GPU buffers
    histogram = torch.zeros(128, dtype=torch.int32, device='cuda')
    bucket_offsets = torch.empty(129, dtype=torch.int32, device='cuda')
    scatter_buf = torch.empty(N, dtype=torch.int32, device='cuda')
    seg_begin = torch.empty(128, dtype=torch.int32, device='cuda')
    seg_end = torch.empty(128, dtype=torch.int32, device='cuda')

    sort_module.sort_cuda(
        input_tensor, output_tensor,
        splitters_tensor, histogram, bucket_offsets, scatter_buf,
        seg_begin, seg_end,
    )
    return output_tensor