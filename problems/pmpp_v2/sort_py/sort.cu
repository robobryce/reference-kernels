#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>

static uint8_t* d_temp_storage = nullptr;
static size_t temp_storage_bytes = 0;

extern "C" {

int sort_float32_init() {
    if (d_temp_storage != nullptr) return 0;

    const int64_t max_n = 100000000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        max_n, 0, 32);

    // Add 10% headroom
    temp_storage_bytes = (temp_storage_bytes * 11 + 9) / 10;

    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    return (err == cudaSuccess) ? 0 : 1;
}

int sort_float32(float* d_in, float* d_out, int n, void* stream_ptr) {
    int ret = sort_float32_init();
    if (ret != 0) return ret;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    const int32_t* key_in = reinterpret_cast<const int32_t*>(d_in);
    int32_t* key_out = reinterpret_cast<int32_t*>(d_out);

    size_t tmp = temp_storage_bytes;
    cudaError_t err = cub::DeviceRadixSort::SortKeys(
        d_temp_storage, tmp,
        key_in, key_out, static_cast<int64_t>(n),
        0, 32,
        stream);

    return (err == cudaSuccess) ? 0 : 3;
}

void sort_float32_cleanup() {
    if (d_temp_storage != nullptr) {
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
    }
}

}  // extern "C"