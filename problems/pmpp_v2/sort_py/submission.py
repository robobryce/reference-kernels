"""
SAMPLE SORT v4: floor-based binning for guaranteed correctness.
For positive IEEE 754 floats, floor(val) is monotonic.
bin = (int32_t)(val) - (int32_t)(min_val), giving rows bins.
But with 256 bins: bin = ((int32_t)val - ioffset) >> shift,
where shift = ceil(log2((int32_t)(max_val - min_val + 1) / 256)).
This produces ~256 monotonic bins. Then segmented sort on lower bits.

This approach eliminates floating-point precision issues:
bin is computed from integer floor(val), not float arithmetic.
The same integer formula is used in both histogram and scatter.

Leaderboard-safe: no streams, no CUDAContext.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cstdint>
#include <cmath>
#include <cfloat>

static torch::Tensor persistent_temp;
static size_t persistent_temp_bytes = 0;
static const int MAX_BINS = 256;

void temp_query_scan(size_t& temp_bytes) {
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
        static_cast<const int*>(nullptr), static_cast<int*>(nullptr), MAX_BINS);
}

void temp_query_segsort(int64_t N, size_t& temp_bytes) {
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr, temp_bytes,
        static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
        N, MAX_BINS,
        static_cast<const int*>(nullptr), static_cast<const int*>(nullptr),
        0, 32);
}

void init_persistent_temp() {
    int64_t max_n = 100'000'000;
    size_t s_bytes = 0, g_bytes = 0;
    temp_query_scan(s_bytes);
    temp_query_segsort(max_n, g_bytes);
    persistent_temp_bytes = std::max(s_bytes, g_bytes);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

// Histogram kernel: bin = ((int)val - offset) >> shift
// For positive IEEE 754 floats, (int)val is floor(val) (truncation toward zero = floor)
__global__ void histogram_kernel(
    const float* __restrict__ input, int N,
    int offset, int shift,
    int* __restrict__ histogram)
{
    __shared__ int s_hist[256];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += gridDim.x * blockDim.x) {
        int ival = static_cast<int>(input[idx]);  // floor for positive values
        int bin = (ival - offset) >> shift;
        if (bin < 0) bin = 0;
        if (bin >= 256) bin = 255;
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x) {
        if (s_hist[i] > 0) atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Scatter kernel: same binning formula, same integer arithmetic
__global__ void scatter_kernel(
    const float* __restrict__ input, int N,
    int offset, int shift,
    int* __restrict__ positions,
    float* __restrict__ scatter_out)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        float val = input[idx];
        int ival = static_cast<int>(val);
        int bin = (ival - offset) >> shift;
        if (bin < 0) bin = 0;
        if (bin >= 256) bin = 255;
        int pos = atomicAdd(&positions[bin], 1);
        scatter_out[pos] = val;
    }
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    int N = static_cast<int>(num_items);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto histogram = torch::zeros({MAX_BINS}, options_int);
    auto offsets = torch::empty({MAX_BINS + 1}, options_int);
    auto positions = torch::empty({MAX_BINS}, options_int);
    auto scatter_buf = torch::empty({N}, options_float);

    // Find min/max using DeviceReduce on float values
    size_t red_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, red_bytes,
        static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
        num_items, cub::Min(), FLT_MAX);
    auto red_temp = torch::empty({static_cast<int64_t>(red_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto d_min = torch::empty({1}, options_float);
    auto d_max = torch::empty({1}, options_float);

    cub::DeviceReduce::Reduce(red_temp.data_ptr(), red_bytes,
        input.const_data_ptr<float>(), d_min.data_ptr<float>(),
        num_items, cub::Min(), FLT_MAX);
    cub::DeviceReduce::Reduce(red_temp.data_ptr(), red_bytes,
        input.const_data_ptr<float>(), d_max.data_ptr<float>(),
        num_items, cub::Max(), -FLT_MAX);
    cudaDeviceSynchronize();

    float h_min, h_max;
    cudaMemcpy(&h_min, d_min.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);

    // Compute binning parameters from floor values
    int imin = static_cast<int>(h_min);
    int imax = static_cast<int>(h_max);
    int irange = imax - imin + 1;
    if (irange < 1) irange = 1;

    // Compute shift to get ~256 bins
    // We want 2^shift ≈ range / 256
    int total_bits = 0;
    int t = irange;
    while (t > 1) { t >>= 1; total_bits++; }
    int shift = total_bits - 8;  // floor(log2(range)) - 8
    if (shift < 0) shift = 0;
    if (shift > 24) shift = 24;

    int offset = imin;

    // Phase 1: Histogram
    {
        int threads = 256;
        int smem = MAX_BINS * sizeof(int);
        // Use enough blocks to saturate the GPU (B200 has ~128 SMs)
        histogram_kernel<<<160, threads, smem>>>(
            input.const_data_ptr<float>(), N,
            offset, shift,
            histogram.data_ptr<int>());
        cudaDeviceSynchronize();
    }

    // Phase 2: Exclusive sum
    {
        size_t temp_bytes = persistent_temp_bytes;
        cub::DeviceScan::ExclusiveSum(
            persistent_temp.data_ptr(), temp_bytes,
            histogram.const_data_ptr<int>(),
            offsets.data_ptr<int>(),
            MAX_BINS);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(positions.data_ptr<int>(), offsets.data_ptr<int>(),
               MAX_BINS * sizeof(int), cudaMemcpyDeviceToDevice);

    // Phase 3: Scatter
    {
        scatter_kernel<<<160, 256>>>(
            input.const_data_ptr<float>(), N,
            offset, shift,
            positions.data_ptr<int>(),
            scatter_buf.data_ptr<float>());
        cudaDeviceSynchronize();
    }

    // Phase 4: Segmented radix sort (32-bit to handle all bits correctly)
    // Even though top bits are binned, within-bin elements can differ in all 32 bits
    {
        size_t temp_bytes = persistent_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortKeys(
            persistent_temp.data_ptr(), temp_bytes,
            scatter_buf.const_data_ptr<float>(),
            output.data_ptr<float>(),
            num_items, MAX_BINS,
            offsets.const_data_ptr<int>(),
            positions.const_data_ptr<int>(),
            0, 32);
        cudaDeviceSynchronize();
    }

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sample_sort_v4_floor_bins',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sample sort: bin by floor(value) >> shift, histogram, scatter,
    then segmented radix sort. Uses integer floor for monotonic binning.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor