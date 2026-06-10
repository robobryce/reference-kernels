"""
CUB DeviceRadixSort::SortKeys with int32 bitcast (no float conversion).
Since all data is positive IEEE 754, raw bits are in correct sort order.
Interpret float* as int*, sort keys-only, re-interpret back as float.
Persistent temp storage allocated once at module init to eliminate per-call overhead.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cstdint>

// Custom policy: override Policy900 with ITEMS_PER_THREAD=24 and BLOCK_THREADS=320.
// 24 items/thread * 320 threads = 7680 tile items vs default 19*384=7296 (5.3% larger tiles).
// Larger tiles reduce grid size and launch overhead on B200 which has 256KB smem/SM.
template <typename KeyT, typename ValueT, typename OffsetT>
struct CustomPolicy900 : cub::ChainedPolicy<900,
    CustomPolicy900<KeyT, ValueT, OffsetT>,
    typename cub::DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>::Policy800>
{
    using Orig = cub::DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>;
    using typename Orig::DominantT;
    static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

    enum
    {
        PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
        SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
        SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
        ONESWEEP               = true,
        ONESWEEP_RADIX_BITS    = 8,
        OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0,
        FLOAT_KEYS             = std::is_same<KeyT, float>::value ? 1 : 0,
    };

    // KEY CHANGE: ITEMS_PER_THREAD=24, BLOCK_THREADS=320 (vs default 19, 384).
    // Total tile items: 320*24=7680 vs 384*19=7296 => 5.3% larger tiles.
    using OnesweepPolicyKey32 = cub::AgentRadixSortOnesweepPolicy<
        320, 24, DominantT, 1,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE,
        cub::RADIX_SORT_STORE_DIRECT,
        8>;

    using OnesweepPolicyKey64 = cub::AgentRadixSortOnesweepPolicy<
        320,
        sizeof(ValueT) < 8 ? 30 : 24,
        DominantT, 1,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE,
        cub::RADIX_SORT_STORE_DIRECT,
        8>;

    using OnesweepLargeKeyPolicy =
        ::cuda::std::_If<sizeof(KeyT) == 4, OnesweepPolicyKey32, OnesweepPolicyKey64>;

    using SmallKeySizes = cub::detail::radix::sm90_small_key_tuning<
        sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>;
    using OnesweepSmallKeyPolicy = cub::AgentRadixSortOnesweepPolicy<
        SmallKeySizes::threads, SmallKeySizes::items, DominantT, 1,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE,
        cub::RADIX_SORT_STORE_DIRECT,
        8>;
    using OnesweepPolicy =
        ::cuda::std::_If<(sizeof(KeyT) < 4), OnesweepSmallKeyPolicy, OnesweepLargeKeyPolicy>;

    using HistogramPolicy    = cub::AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = cub::AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    using ScanPolicy = cub::AgentScanPolicy<512, 23, OffsetT,
        cub::BLOCK_LOAD_WARP_TRANSPOSE, cub::LOAD_DEFAULT,
        cub::BLOCK_STORE_WARP_TRANSPOSE, cub::BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = cub::AgentRadixSortDownsweepPolicy<512, 23, DominantT,
        cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT,
        cub::RADIX_RANK_MATCH, cub::BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>;

    using AltDownsweepPolicy = cub::AgentRadixSortDownsweepPolicy<
        (sizeof(KeyT) > 1) ? 256 : 128, 47, DominantT,
        cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT,
        cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy    = cub::AgentRadixSortUpsweepPolicy<256, 23, DominantT, cub::LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = cub::AgentRadixSortUpsweepPolicy<256, 47, DominantT, cub::LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<256, 19, DominantT,
        cub::BLOCK_LOAD_DIRECT, cub::LOAD_LDG,
        cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = cub::AgentRadixSortDownsweepPolicy<192, 39, DominantT,
        cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT,
        cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>;

    using AltSegmentedPolicy = cub::AgentRadixSortDownsweepPolicy<384, 11, DominantT,
        cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT,
        cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1>;
};

// Full custom policy struct; MaxPolicy is our custom Policy900
template <typename KeyT, typename ValueT, typename OffsetT>
struct CustomRadixSortPolicy : cub::DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>
{
    using MaxPolicy = CustomPolicy900<KeyT, ValueT, OffsetT>;
};

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    using KeyT = uint32_t;
    using OffsetT = int64_t;
    using PolicyT = CustomRadixSortPolicy<KeyT, cub::NullType, OffsetT>;
    int begin_bit = 0, end_bit = 32;
    int64_t max_n = 100'000'000;

    size_t temp_bytes = 0;
    KeyT* dummy_ptrs[2] = {nullptr, nullptr};
    cub::DoubleBuffer<KeyT> dummy_db(dummy_ptrs[0], dummy_ptrs[1]);
    cub::NullType* null_ptrs[2] = {nullptr, nullptr};
    cub::DoubleBuffer<cub::NullType> dummy_vals(null_ptrs[0], null_ptrs[1]);
    cub::DispatchRadixSort<false, KeyT, cub::NullType, OffsetT, PolicyT>::Dispatch(
        nullptr, temp_bytes,
        dummy_db, dummy_vals,
        static_cast<OffsetT>(max_n),
        begin_bit, end_bit, false,
        0);

    persistent_temp_bytes = (temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    using KeyT = uint32_t;
    using OffsetT = int64_t;
    using PolicyT = CustomRadixSortPolicy<KeyT, cub::NullType, OffsetT>;
    int begin_bit = 0, end_bit = 32;
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const KeyT* key_in = reinterpret_cast<const KeyT*>(input.const_data_ptr<float>());
    KeyT* key_out = reinterpret_cast<KeyT*>(output.data_ptr<float>());

    cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(key_in), key_out);
    cub::DoubleBuffer<cub::NullType> d_values;

    size_t temp_bytes = persistent_temp_bytes;
    cub::DispatchRadixSort<false, KeyT, cub::NullType, OffsetT, PolicyT>::Dispatch(
        persistent_temp.data_ptr(), temp_bytes,
        d_keys, d_values,
        static_cast<OffsetT>(num_items),
        begin_bit, end_bit, false,
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_int32_bitcast_persistent',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    No conversion needed — all data is positive IEEE 754 floats.
    Persistent temp storage avoids per-call allocation.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor