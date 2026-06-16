// CUDA quad Ocean Monument structure-position candidate search.
//
// This ports cubiomes' Monument getLargeStructurePos() structure-position
// formula to CUDA. It does not perform biome viability checks, because that
// requires porting the cubiomes biome generator/layers to GPU.
//
// Build: nvcc -O3 -std=c++17 -o ocean_monument_quad_cuda ocean_monument_quad_cuda.cu

#include <cuda_runtime.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "biome_noise_cuda.h"

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;
static const int MONUMENT_SALT = 10387313;
static const int MONUMENT_REGION_SIZE = 32;
static const int MONUMENT_CHUNK_RANGE = 27;

struct MonumentBox {
    int pos_x;
    int pos_z;
    int center_x;
    int center_z;
    int min_x;
    int min_z;
    int max_x;
    int max_z;
};

struct QuadCandidate {
    int64_t seed;
    int center_x[4];
    int center_z[4];
    int window_chunk_x;
    int window_chunk_z;
    int window_min_x;
    int window_min_z;
    int window_max_x;
    int window_max_z;
};

__device__ __host__ static inline int floor_div_i(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r && ((r > 0) != (b > 0))) q--;
    return q;
}

__device__ __host__ static inline int ceil_div_i(int a, int b) {
    return -floor_div_i(-a, b);
}

__device__ __host__ static inline int max_i(int a, int b) {
    return a > b ? a : b;
}

__device__ __host__ static inline int min_i(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ uint64_t next_seed(uint64_t seed) {
    return (seed * JAVA_MULT + JAVA_ADD) & JAVA_MASK;
}

__device__ __forceinline__ int next31_mod(uint64_t *seed, int bound) {
    *seed = next_seed(*seed);
    return (int)(*seed >> 17) % bound;
}

__device__ __forceinline__ void monument_pos(uint64_t world_seed, int rx, int rz, int *out_x, int *out_z) {
    uint64_t s = world_seed + (uint64_t)((int64_t)rx * 341873128712LL) + (uint64_t)((int64_t)rz * 132897987541LL) + (uint64_t)MONUMENT_SALT;
    s = (s ^ JAVA_MULT) & JAVA_MASK;
    int x = next31_mod(&s, MONUMENT_CHUNK_RANGE);
    x += next31_mod(&s, MONUMENT_CHUNK_RANGE);
    int z = next31_mod(&s, MONUMENT_CHUNK_RANGE);
    z += next31_mod(&s, MONUMENT_CHUNK_RANGE);
    x >>= 1;
    z >>= 1;
    *out_x = (rx * MONUMENT_REGION_SIZE + x) << 4;
    *out_z = (rz * MONUMENT_REGION_SIZE + z) << 4;
}

__device__ __forceinline__ MonumentBox make_box(int pos_x, int pos_z, int footprint_blocks, int center_offset_blocks) {
    int half_before = footprint_blocks / 2;
    int half_after = footprint_blocks - half_before - 1;
    int cx = pos_x + center_offset_blocks;
    int cz = pos_z + center_offset_blocks;
    MonumentBox box;
    box.pos_x = pos_x;
    box.pos_z = pos_z;
    box.center_x = cx;
    box.center_z = cz;
    box.min_x = cx - half_before;
    box.min_z = cz - half_before;
    box.max_x = cx + half_after;
    box.max_z = cz + half_after;
    return box;
}

__device__ __forceinline__ bool containing_window_for_quad(
    const MonumentBox *m,
    int window_chunks,
    int radius,
    bool window_inside_radius,
    int *window_chunk_x,
    int *window_chunk_z
) {
    int window_blocks = window_chunks * 16;
    int min_x = m[0].min_x, min_z = m[0].min_z;
    int max_x = m[0].max_x, max_z = m[0].max_z;
    for (int i = 1; i < 4; i++) {
        min_x = min_i(min_x, m[i].min_x);
        min_z = min_i(min_z, m[i].min_z);
        max_x = max_i(max_x, m[i].max_x);
        max_z = max_i(max_z, m[i].max_z);
    }

    int sx0 = ceil_div_i(max_x - (window_blocks - 1), 16);
    int sx1 = floor_div_i(min_x, 16);
    int sz0 = ceil_div_i(max_z - (window_blocks - 1), 16);
    int sz1 = floor_div_i(min_z, 16);
    if (window_inside_radius) {
        sx0 = max_i(sx0, ceil_div_i(-radius, 16));
        sz0 = max_i(sz0, ceil_div_i(-radius, 16));
        sx1 = min_i(sx1, floor_div_i(radius - (window_blocks - 1), 16));
        sz1 = min_i(sz1, floor_div_i(radius - (window_blocks - 1), 16));
    }
    if (sx0 > sx1 || sz0 > sz1) return false;
    *window_chunk_x = sx0;
    *window_chunk_z = sz0;
    return true;
}

__global__ void quad_monument_kernel(
    int64_t start,
    int64_t count,
    int reg_min,
    int reg_span,
    int radius,
    int window_chunks,
    int footprint_blocks,
    int center_offset_blocks,
    bool window_inside_radius,
    QuadCandidate *candidates,
    unsigned int *candidate_count,
    unsigned int max_output
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int64_t seed = start + idx;
    uint64_t u_seed = (uint64_t)seed;
    for (int rz_i = 0; rz_i < reg_span; rz_i++) {
        int rz = reg_min + rz_i;
        for (int rx_i = 0; rx_i < reg_span; rx_i++) {
            int rx = reg_min + rx_i;
            int x0, z0, x1, z1, x2, z2, x3, z3;
            monument_pos(u_seed, rx, rz, &x0, &z0);
            monument_pos(u_seed, rx + 1, rz, &x1, &z1);
            monument_pos(u_seed, rx, rz + 1, &x2, &z2);
            monument_pos(u_seed, rx + 1, rz + 1, &x3, &z3);

            MonumentBox m[4];
            m[0] = make_box(x0, z0, footprint_blocks, center_offset_blocks);
            m[1] = make_box(x1, z1, footprint_blocks, center_offset_blocks);
            m[2] = make_box(x2, z2, footprint_blocks, center_offset_blocks);
            m[3] = make_box(x3, z3, footprint_blocks, center_offset_blocks);
            int wcx = 0, wcz = 0;
            if (!containing_window_for_quad(m, window_chunks, radius, window_inside_radius, &wcx, &wcz)) continue;

            unsigned int slot = atomicAdd(candidate_count, 1U);
            if (slot < max_output) {
                QuadCandidate *q = &candidates[slot];
                q->seed = seed;
                for (int i = 0; i < 4; i++) {
                    q->center_x[i] = m[i].center_x;
                    q->center_z[i] = m[i].center_z;
                }
                q->window_chunk_x = wcx;
                q->window_chunk_z = wcz;
                q->window_min_x = wcx * 16;
                q->window_min_z = wcz * 16;
                q->window_max_x = (wcx + window_chunks) * 16 - 1;
                q->window_max_z = (wcz + window_chunks) * 16 - 1;
            }
            return;
        }
    }
}

__global__ void init_candidate_biome_noise_kernel(BiomeNoiseCuda *bn, const QuadCandidate *candidates, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_biome_noise_cuda(&bn[idx]);
    set_biome_seed_cuda(&bn[idx], (uint64_t)candidates[idx].seed);
}

__global__ void monument_viability_kernel(
    const QuadCandidate *candidates,
    const BiomeNoiseCuda *bn,
    int count,
    int center_offset_blocks,
    uint8_t *viable
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const QuadCandidate *q = &candidates[idx];
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        int pos_x = q->center_x[i] - center_offset_blocks;
        int pos_z = q->center_z[i] - center_offset_blocks;
        if (!monument_viable_118_cuda(&bn[idx], pos_x, pos_z)) {
            ok = false;
            break;
        }
    }
    viable[idx] = ok ? 1 : 0;
}

static void cuda_check(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static long long parse_ll_arg(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --start SEED --count N --out quad_monuments.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N          origin-centered square +/-N blocks (default 60000)\n"
        "  --window-inside-radius     require the full 17x17 window inside +/-radius\n"
        "  --cluster-window-chunks N  chunk-aligned cluster square size (default 17)\n"
        "  --required-count N         currently only 4 is supported (default 4)\n"
        "  --footprint-blocks N       monument footprint width/depth to contain (default 58)\n"
        "  --center-offset-blocks N   center offset from structure pos (default 8)\n"
        "  --skip-biome-viability     only check structure positions, not monument ocean viability\n"
        "  --max-candidates N         stop after N hits, 0 = no cap\n"
        "  --batch-size N             seeds per GPU launch (default 1048576)\n"
        "  --threads N                CUDA threads per block (default 256)\n",
        argv0);
}

int main(int argc, char **argv) {
    int64_t start = 0;
    int64_t count = -1;
    int radius = 60000;
    int window_chunks = 17;
    int required_count = 4;
    int footprint_blocks = 58;
    int center_offset_blocks = 8;
    int64_t max_candidates = 0;
    int64_t batch_size = 1048576;
    int threads = 256;
    bool window_inside_radius = false;
    bool check_biome_viability = true;
    const char *out_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_ll_arg(argv[++i], "start");
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = parse_ll_arg(argv[++i], "count");
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_ll_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--cluster-window-chunks") && i + 1 < argc) window_chunks = (int)parse_ll_arg(argv[++i], "cluster-window-chunks");
        else if (!strcmp(argv[i], "--required-count") && i + 1 < argc) required_count = (int)parse_ll_arg(argv[++i], "required-count");
        else if (!strcmp(argv[i], "--footprint-blocks") && i + 1 < argc) footprint_blocks = (int)parse_ll_arg(argv[++i], "footprint-blocks");
        else if (!strcmp(argv[i], "--center-offset-blocks") && i + 1 < argc) center_offset_blocks = (int)parse_ll_arg(argv[++i], "center-offset-blocks");
        else if (!strcmp(argv[i], "--max-candidates") && i + 1 < argc) max_candidates = parse_ll_arg(argv[++i], "max-candidates");
        else if (!strcmp(argv[i], "--batch-size") && i + 1 < argc) batch_size = parse_ll_arg(argv[++i], "batch-size");
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (int)parse_ll_arg(argv[++i], "threads");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) i++; // structure positions are version-stable for monuments here
        else if (!strcmp(argv[i], "--window-inside-radius")) window_inside_radius = true;
        else if (!strcmp(argv[i], "--skip-biome-viability")) check_biome_viability = false;
        else { usage(argv[0]); return 2; }
    }
    if (count < 0 || !out_path) { usage(argv[0]); return 2; }
    if (required_count != 4) {
        fprintf(stderr, "Only --required-count 4 is currently supported.\n");
        return 2;
    }
    if (radius <= 0 || window_chunks <= 0 || footprint_blocks <= 0 || batch_size <= 0 || threads <= 0) {
        fprintf(stderr, "radius/window/footprint/batch/threads must be positive.\n");
        return 2;
    }

    int region_blocks = MONUMENT_REGION_SIZE * 16;
    int margin = window_chunks * 16 + footprint_blocks + region_blocks;
    int reg_min = floor_div_i(-radius - margin, region_blocks) - 1;
    int reg_max = floor_div_i(radius + margin, region_blocks) + 1;
    int reg_span = reg_max - reg_min;

    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); return 1; }
    fprintf(out,
        "seed,monument1_center_x,monument1_center_z,monument2_center_x,monument2_center_z,"
        "monument3_center_x,monument3_center_z,monument4_center_x,monument4_center_z,"
        "quad_window_chunk_x,quad_window_chunk_z,quad_window_min_block_x,quad_window_min_block_z,"
        "quad_window_max_block_x,quad_window_max_block_z,cluster_window_chunks,search_radius_blocks,"
        "mc_approx,monument_footprint_blocks,monument_center_offset_blocks,quad_backend,biome_viability_checked\n");

    QuadCandidate *d_candidates = NULL;
    unsigned int *d_candidate_count = NULL;
    cuda_check(cudaMalloc((void **)&d_candidates, (size_t)batch_size * sizeof(QuadCandidate)), "cudaMalloc candidates");
    cuda_check(cudaMalloc((void **)&d_candidate_count, sizeof(unsigned int)), "cudaMalloc candidate_count");
    QuadCandidate *h_candidates = (QuadCandidate *)malloc((size_t)batch_size * sizeof(QuadCandidate));
    uint8_t *h_viable = (uint8_t *)malloc((size_t)batch_size * sizeof(uint8_t));
    if (!h_candidates) { perror("malloc candidates"); return 1; }
    if (!h_viable) { perror("malloc viability"); return 1; }

    int64_t scanned = 0;
    int64_t total_candidates = 0;
    int64_t total_structure_candidates = 0;
    while (scanned < count) {
        int64_t todo = count - scanned;
        int64_t batch64 = todo < batch_size ? todo : batch_size;
        unsigned int batch = (unsigned int)batch64;
        cuda_check(cudaMemset(d_candidate_count, 0, sizeof(unsigned int)), "cudaMemset candidate_count");
        int blocks = (int)((batch64 + threads - 1) / threads);
        quad_monument_kernel<<<blocks, threads>>>(
            start + scanned,
            batch64,
            reg_min,
            reg_span,
            radius,
            window_chunks,
            footprint_blocks,
            center_offset_blocks,
            window_inside_radius,
            d_candidates,
            d_candidate_count,
            batch);
        cuda_check(cudaGetLastError(), "quad_monument_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "quad_monument_kernel sync");

        unsigned int found = 0;
        cuda_check(cudaMemcpy(&found, d_candidate_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy candidate_count");
        unsigned int capped = found > batch ? batch : found;
        total_structure_candidates += found;
        if (capped > 0) {
            if (check_biome_viability) {
                BiomeNoiseCuda *d_bn = NULL;
                uint8_t *d_viable = NULL;
                cuda_check(cudaMalloc((void **)&d_bn, (size_t)capped * sizeof(BiomeNoiseCuda)), "cudaMalloc biome noise candidates");
                cuda_check(cudaMalloc((void **)&d_viable, (size_t)capped * sizeof(uint8_t)), "cudaMalloc viability flags");
                int viability_blocks = (int)((capped + threads - 1) / threads);
                init_candidate_biome_noise_kernel<<<viability_blocks, threads>>>(d_bn, d_candidates, (int)capped);
                cuda_check(cudaGetLastError(), "init_candidate_biome_noise_kernel launch");
                cuda_check(cudaDeviceSynchronize(), "init_candidate_biome_noise_kernel sync");
                monument_viability_kernel<<<viability_blocks, threads>>>(d_candidates, d_bn, (int)capped, center_offset_blocks, d_viable);
                cuda_check(cudaGetLastError(), "monument_viability_kernel launch");
                cuda_check(cudaDeviceSynchronize(), "monument_viability_kernel sync");
                cuda_check(cudaMemcpy(h_viable, d_viable, (size_t)capped * sizeof(uint8_t), cudaMemcpyDeviceToHost), "copy viability flags");
                cudaFree(d_bn);
                cudaFree(d_viable);
            } else {
                memset(h_viable, 1, (size_t)capped * sizeof(uint8_t));
            }
            cuda_check(cudaMemcpy(h_candidates, d_candidates, (size_t)capped * sizeof(QuadCandidate), cudaMemcpyDeviceToHost), "copy candidates");
            for (unsigned int i = 0; i < capped; i++) {
                if (!h_viable[i]) continue;
                if (max_candidates > 0 && total_candidates >= max_candidates) break;
                QuadCandidate *q = &h_candidates[i];
                fprintf(out,
                    "%" PRId64 ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,26.2,%d,%d,%s,%s\n",
                    q->seed,
                    q->center_x[0], q->center_z[0],
                    q->center_x[1], q->center_z[1],
                    q->center_x[2], q->center_z[2],
                    q->center_x[3], q->center_z[3],
                    q->window_chunk_x, q->window_chunk_z,
                    q->window_min_x, q->window_min_z,
                    q->window_max_x, q->window_max_z,
                    window_chunks, radius, footprint_blocks, center_offset_blocks,
                    check_biome_viability ? "cuda_structure_biome_26_2" : "cuda_structure",
                    check_biome_viability ? "true" : "false");
                total_candidates++;
            }
        }
        scanned += batch64;
        fprintf(stderr, "progress: start=%" PRId64 " scanned=%" PRId64 "/%" PRId64 " structure_candidates=%" PRId64 " candidates=%" PRId64 " backend=%s\n",
            start, scanned, count, total_structure_candidates, total_candidates,
            check_biome_viability ? "cuda_structure_biome_26_2" : "cuda_structure");
        fflush(stderr);
        if (max_candidates > 0 && total_candidates >= max_candidates) break;
    }

    fclose(out);
    free(h_candidates);
    free(h_viable);
    cudaFree(d_candidates);
    cudaFree(d_candidate_count);
    fprintf(stderr, "done: scanned=%" PRId64 " structure_candidates=%" PRId64 " candidates=%" PRId64 " output=%s backend=%s\n",
        scanned, total_structure_candidates, total_candidates, out_path,
        check_biome_viability ? "cuda_structure_biome_26_2" : "cuda_structure");
    return 0;
}
