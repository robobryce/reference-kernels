import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# Phase 1: BlockRadixSort per chunk (256 threads, 8 items/thread, 2048 items per chunk)
# Phase 2: Sequential pair merge per level (one kernel launch per level)

cpp_source = """
#include <torch/extension.h>
torch::Tensor chunk_sort_merge(torch::Tensor input, torch::Tensor output);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Phase 1: BlockRadixSort on 2048-item chunks
// chunk_offsets[j] = start position of chunk j, num_chunks total.

__global__ void sort_chunks(
    float* __restrict__ data,
    const int* __restrict__ chunk_offsets,  // [num_chunks]
    int num_chunks)
{
    int c = blockIdx.x;
    if (c >= num_chunks) return;
    int start = chunk_offsets[c];
    int end   = (c + 1 < num_chunks) ? chunk_offsets[c + 1] : start + 2048;
    // Actually we pass correct sizes via data; just use fixed 2048 with INFINITY pad.
    // Simpler: use chunk sizes array.

    // For now, treat all chunks as exactly 2048 items (pad with INFINITY)
    // We'll handle partial chunks in the host code by rounding up
}

// Phase 2 merge kernel — use templated items_per_block for GPU access
template<int ITEMS_PER_BLOCK>
__global__ void merge_level_t(
    const float* __restrict__ src,
    const int*    __restrict__ pair_info,
    float*        __restrict__ dst,
    const int*    __restrict__ dst_offs,
    int num_pairs)
{
    int p = blockIdx.y;
    if (p >= num_pairs) return;
    int a0 = pair_info[4*p], aL = pair_info[4*p+1], b0 = pair_info[4*p+2], bL = pair_info[4*p+3];
    int tot = aL + bL, d0 = dst_offs[p];
    int bs = (int)blockIdx.x * ITEMS_PER_BLOCK;
    if (bs >= tot) return;
    int bl = min(ITEMS_PER_BLOCK, tot - bs);
    constexpr int IPT = ITEMS_PER_BLOCK / 256;
    int tid = (int)threadIdx.x;
    int ts = bs + tid * IPT, te = min(ts + IPT, bs + bl);
    if (ts >= te) return; int tl = te - ts;
    int diag = ts, lo = max(0, diag - bL), hi = min(diag, aL);
    while (lo < hi) {
        int m = (lo+hi)>>1, mB = diag-1-m;
        if (mB < 0) hi = m; else if (mB >= bL) lo = m+1;
        else if (src[a0+m] <= src[b0+mB]) lo = m+1; else hi = m;
    }
    int pA = a0 + lo, pB = b0 + diag - lo, aE = a0 + aL, bE = b0 + bL;
    float* out = dst + d0 + ts;
    for (int i = 0; i < tl; i++) {
        if (pA < aE && (pB >= bE || src[pA] <= src[pB])) out[i] = src[pA++]; else out[i] = src[pB++];
    }
}

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

torch::Tensor chunk_sort_merge(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) { if(N==1) output[0]=input[0]; return output; }

    int rows = (int)std::sqrt((double)N);
    int cols = (N + rows - 1) / rows;

    // Row offsets
    std::vector<int> ro(rows + 1);
    for (int i = 0; i < rows; i++) ro[i] = std::min(i * cols, N);
    ro[rows] = N;
    int eff = rows;
    while (eff > 0 && ro[eff-1] == ro[eff]) eff--;
    if (eff == 0) return output;

    // Chunk layout: each row is one chunk_spec for Phase 1
    // Phase 1: sort each row as a segment (use DeviceSegmentedRadixSort)
    {
        auto d_ro = torch::empty({eff + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        cudaMemcpy(d_ro.data_ptr<int>(), ro.data(), (eff+1)*sizeof(int), cudaMemcpyHostToDevice);
        size_t tb = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, tb,
            input.data_ptr<float>(), output.data_ptr<float>(), N, eff,
            d_ro.data_ptr<int>(), d_ro.data_ptr<int>()+1, 0, sizeof(float)*8);
        auto dtmp = torch::empty({(int64_t)tb}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cub::DeviceSegmentedRadixSort::SortKeys(dtmp.data_ptr(), tb,
            input.data_ptr<float>(), output.data_ptr<float>(), N, eff,
            d_ro.data_ptr<int>(), d_ro.data_ptr<int>()+1, 0, sizeof(float)*8);
        cudaDeviceSynchronize();
    }
    // Copy output -> input
    { int blk=(N+255)/256; copy_kernel<<<blk,256>>>(output.data_ptr<float>(), input.data_ptr<float>(), N); cudaDeviceSynchronize(); }

    // Phase 2: merge tree
    // Build all level metadata and upload once
    int total_pd=0, total_do=0;
    std::vector<std::vector<int>> all_pd, all_do;
    std::vector<int> all_np, all_mc;
    std::vector<int> all_ol, all_os, all_od;

    {
        std::vector<int> s(ro.begin(), ro.begin()+eff+1);
        std::vector<int> l(eff);
        for (int i=0;i<eff;i++) l[i]=s[i+1]-s[i];
        int ns=eff;
        while (ns>1) {
            int np=ns/2, pos=0, mc=0;
            std::vector<int> pd, ds;
            for (int i=0;i<np;i++) {
                pd.insert(pd.end(), {s[2*i], l[2*i], s[2*i+1], l[2*i+1]});
                ds.push_back(pos);
                int tot=l[2*i]+l[2*i+1]; pos+=tot;
                int ck=(tot+2047)/2048; if(ck>mc) mc=ck;
            }
            int ol=0,os=0,od=0;
            if (ns%2 && ns>1) { ol=l[ns-1]; os=s[ns-1]; od=pos; pos+=ol; }
            all_pd.push_back(pd); all_do.push_back(ds);
            all_np.push_back(np); all_mc.push_back(mc);
            all_ol.push_back(ol); all_os.push_back(os); all_od.push_back(od);
            total_pd += (int)pd.size(); total_do += (int)ds.size();

            std::vector<int> ns_s, ns_l;
            pos=0;
            for (int i=0;i<np;i++) { ns_s.push_back(pos); ns_l.push_back(l[2*i]+l[2*i+1]); pos+=l[2*i]+l[2*i+1]; }
            if (ns%2) { ns_s.push_back(pos); ns_l.push_back(l[ns-1]); pos+=l[ns-1]; }
            ns_s.push_back(pos); s=ns_s; l=ns_l; ns=(int)ns_l.size();
        }
    }

    // Upload all data at once
    auto g_pd = torch::empty({total_pd}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto g_do = torch::empty({total_do}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int off_pd=0, off_do=0;
    for (size_t i=0; i<all_pd.size(); i++) {
        if(!all_pd[i].empty()) cudaMemcpy(g_pd.data_ptr<int>()+off_pd, all_pd[i].data(), all_pd[i].size()*sizeof(int), cudaMemcpyHostToDevice);
        if(!all_do[i].empty()) cudaMemcpy(g_do.data_ptr<int>()+off_do, all_do[i].data(), all_do[i].size()*sizeof(int), cudaMemcpyHostToDevice);
        off_pd += (int)all_pd[i].size(); off_do += (int)all_do[i].size();
    }

    float* bufs[2] = { input.data_ptr<float>(), output.data_ptr<float>() };
    int src_i=0, nlev=(int)all_np.size();
    off_pd=0; off_do=0;

    for (int lev=0; lev<nlev; lev++) {
        int np=all_np[lev], mc=all_mc[lev];
        if (np>0) {
            dim3 grid(mc, np);
            merge_level_t<2048><<<grid,256>>>(
                bufs[src_i], g_pd.data_ptr<int>()+off_pd,
                bufs[1-src_i], g_do.data_ptr<int>()+off_do, np);
        }
        if (all_ol[lev]>0) {
            int blk=(all_ol[lev]+255)/256;
            copy_kernel<<<blk,256>>>(bufs[src_i]+all_os[lev], bufs[1-src_i]+all_od[lev], all_ol[lev]);
        }
        cudaDeviceSynchronize();
        src_i = 1-src_i;
        off_pd += (int)all_pd[lev].size();
        off_do += (int)all_do[lev].size();
    }
    cudaDeviceSynchronize();
    if (src_i==0) {
        int blk=(N+255)/256;
        copy_kernel<<<blk,256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
        cudaDeviceSynchronize();
    }
    return output;
}
"""

module = load_inline(
    name='chunk_sort_merge',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['chunk_sort_merge'],
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Phase 1: DeviceSegmentedRadixSort sorts each row independently.
    Phase 2: Full merge tree with single-upload metadata, no per-level allocations.
    """
    input_tensor, output_tensor = data
    module.chunk_sort_merge(input_tensor, output_tensor)
    return output_tensor