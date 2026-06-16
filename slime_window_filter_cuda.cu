// CUDA exact 17x17 slime-window scanner for candidate Minecraft Java seeds.
//
// CPU work here is limited to CSV I/O and kernel orchestration. The slime grid,
// window sums, and max search are computed on the GPU.
//
// Build: nvcc -O3 -std=c++17 -o slime_window_filter_cuda slime_window_filter_cuda.cu

#include <cuda_runtime.h>

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;

struct BestWindow {
    int best;
    int wx;
    int wz;
};

__device__ __forceinline__ int java_next_bits_dev(uint64_t *seed, int bits) {
    *seed = ((*seed * JAVA_MULT) + JAVA_ADD) & JAVA_MASK;
    return (int)(*seed >> (48 - bits));
}

__device__ __forceinline__ int java_next_int_dev(uint64_t *seed, int bound) {
    int bits, val;
    if ((bound & -bound) == bound) {
        return (int)((bound * (int64_t)java_next_bits_dev(seed, 31)) >> 31);
    }
    do {
        bits = java_next_bits_dev(seed, 31);
        val = bits % bound;
    } while (bits - val + (bound - 1) < 0);
    return val;
}

__device__ __forceinline__ int64_t java_int_to_long_dev(uint32_t value) {
    return (value & 0x80000000U) ? (int64_t)value - 0x100000000LL : (int64_t)value;
}

__device__ __forceinline__ uint32_t java_int_mul2_dev(int32_t a, int32_t b) {
    return (uint32_t)a * (uint32_t)b;
}

__device__ __forceinline__ uint32_t java_int_mul3_dev(int32_t a, int32_t b, int32_t c) {
    return ((uint32_t)a * (uint32_t)b) * (uint32_t)c;
}

__device__ __forceinline__ bool is_slime_chunk_dev(int64_t world_seed, int32_t cx, int32_t cz) {
    uint64_t s = (uint64_t)world_seed;
    s += (uint64_t)java_int_to_long_dev(java_int_mul3_dev(cx, cx, 4987142));
    s += (uint64_t)java_int_to_long_dev(java_int_mul2_dev(cx, 5947611));
    s += (uint64_t)(java_int_to_long_dev(java_int_mul2_dev(cz, cz)) * 4392871LL);
    s += (uint64_t)java_int_to_long_dev(java_int_mul2_dev(cz, 389711));
    s ^= 987234911ULL;
    uint64_t rnd = (s ^ JAVA_MULT) & JAVA_MASK;
    return java_next_int_dev(&rnd, 10) == 0;
}

__global__ void fill_slime_grid_kernel(int64_t seed, int chunk_min, int width, uint8_t *grid) {
    int64_t total = (int64_t)width * width;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int x = (int)(idx % width);
        int z = (int)(idx / width);
        grid[idx] = is_slime_chunk_dev(seed, chunk_min + x, chunk_min + z) ? 1 : 0;
    }
}

__global__ void vertical_sums_kernel(const uint8_t *grid, int width, int window_count, int window_chunks, uint16_t *vertical) {
    int64_t total = (int64_t)window_count * width;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int x = (int)(idx % width);
        int wz = (int)(idx / width);
        int sum = 0;
        int base = wz * width + x;
        for (int dz = 0; dz < window_chunks; dz++) {
            sum += grid[base + dz * width];
        }
        vertical[idx] = (uint16_t)sum;
    }
}

__global__ void best_window_kernel(
    const uint16_t *vertical,
    int width,
    int window_count,
    int window_chunks,
    BestWindow *block_bests
) {
    extern __shared__ BestWindow shared[];
    BestWindow local;
    local.best = -1;
    local.wx = 0;
    local.wz = 0;

    int64_t total = (int64_t)window_count * window_count;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int wx = (int)(idx % window_count);
        int wz = (int)(idx / window_count);
        int sum = 0;
        int base = wz * width + wx;
        for (int dx = 0; dx < window_chunks; dx++) {
            sum += vertical[base + dx];
        }
        if (sum > local.best) {
            local.best = sum;
            local.wx = wx;
            local.wz = wz;
        }
    }

    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset && shared[threadIdx.x + offset].best > shared[threadIdx.x].best) {
            shared[threadIdx.x] = shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_bests[blockIdx.x] = shared[0];
    }
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

static int64_t parse_seed_from_csv_line(const char *line) {
    errno = 0;
    char *end = NULL;
    int64_t seed = strtoll(line, &end, 10);
    if (errno || end == line || (*end && *end != ',' && *end != '\n' && *end != '\r')) {
        fprintf(stderr, "Could not parse seed from line: %.120s\n", line);
        exit(2);
    }
    return seed;
}

static int64_t floor_div_i64(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r && ((r > 0) != (b > 0))) q--;
    return q;
}

static int64_t ceil_div_i64(int64_t a, int64_t b) {
    return -floor_div_i64(-a, b);
}

static void trim_line_end(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

static BestWindow find_best_slime_window_cuda(
    int64_t seed,
    int chunk_min,
    int width,
    int window_count,
    int window_chunks,
    int threads,
    int blocks,
    uint8_t *d_grid,
    uint16_t *d_vertical,
    BestWindow *d_block_bests,
    BestWindow *h_block_bests
) {
    fill_slime_grid_kernel<<<blocks, threads>>>(seed, chunk_min, width, d_grid);
    cuda_check(cudaGetLastError(), "fill_slime_grid_kernel launch");
    vertical_sums_kernel<<<blocks, threads>>>(d_grid, width, window_count, window_chunks, d_vertical);
    cuda_check(cudaGetLastError(), "vertical_sums_kernel launch");
    best_window_kernel<<<blocks, threads, (size_t)threads * sizeof(BestWindow)>>>(
        d_vertical, width, window_count, window_chunks, d_block_bests);
    cuda_check(cudaGetLastError(), "best_window_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "slime window kernels");
    cuda_check(cudaMemcpy(h_block_bests, d_block_bests, (size_t)blocks * sizeof(BestWindow), cudaMemcpyDeviceToHost), "copy block bests");

    BestWindow best;
    best.best = -1;
    best.wx = 0;
    best.wz = 0;
    for (int i = 0; i < blocks; i++) {
        if (h_block_bests[i].best > best.best) {
            best = h_block_bests[i];
        }
    }
    return best;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out slime_windows.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N     require full window inside square +/-N blocks (default 60000)\n"
        "  --window-chunks N     chunk-aligned square window size (default 17)\n"
        "  --min-slime-chunks N  keep rows with best count >= N (default 0)\n"
        "  --limit N             process at most N candidate rows, 0 = all\n"
        "  --threads N           CUDA threads per block (default 256)\n"
        "  --blocks N            CUDA block count (default auto, capped at 65535)\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    int radius_blocks = 60000;
    int window_chunks = 17;
    int min_slime_chunks = 0;
    int threads = 256;
    int blocks_arg = 0;
    int64_t limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius_blocks = (int)parse_ll_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--window-chunks") && i + 1 < argc) window_chunks = (int)parse_ll_arg(argv[++i], "window-chunks");
        else if (!strcmp(argv[i], "--min-slime-chunks") && i + 1 < argc) min_slime_chunks = (int)parse_ll_arg(argv[++i], "min-slime-chunks");
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_ll_arg(argv[++i], "limit");
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (int)parse_ll_arg(argv[++i], "threads");
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks_arg = (int)parse_ll_arg(argv[++i], "blocks");
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path) { usage(argv[0]); return 2; }
    if (radius_blocks <= 0 || window_chunks <= 0 || min_slime_chunks < 0 || threads <= 0) {
        fprintf(stderr, "radius/window/threads must be positive and min slime chunks must be non-negative.\n");
        return 2;
    }

    int64_t window_blocks = (int64_t)window_chunks * 16;
    int64_t start_min64 = ceil_div_i64(-(int64_t)radius_blocks, 16);
    int64_t start_max64 = floor_div_i64((int64_t)radius_blocks - (window_blocks - 1), 16);
    if (start_min64 > start_max64) {
        fprintf(stderr, "window does not fit inside radius\n");
        return 2;
    }
    int chunk_min = (int)start_min64;
    int window_count = (int)(start_max64 - start_min64 + 1);
    int width = window_count + window_chunks - 1;
    int64_t total_windows = (int64_t)window_count * window_count;
    int blocks = blocks_arg > 0 ? blocks_arg : (int)((total_windows + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    size_t grid_bytes = (size_t)width * (size_t)width * sizeof(uint8_t);
    size_t vertical_bytes = (size_t)window_count * (size_t)width * sizeof(uint16_t);
    uint8_t *d_grid = NULL;
    uint16_t *d_vertical = NULL;
    BestWindow *d_block_bests = NULL;
    cuda_check(cudaMalloc((void **)&d_grid, grid_bytes), "cudaMalloc grid");
    cuda_check(cudaMalloc((void **)&d_vertical, vertical_bytes), "cudaMalloc vertical");
    cuda_check(cudaMalloc((void **)&d_block_bests, (size_t)blocks * sizeof(BestWindow)), "cudaMalloc block bests");
    BestWindow *h_block_bests = (BestWindow *)malloc((size_t)blocks * sizeof(BestWindow));
    if (!h_block_bests) { perror("malloc block bests"); return 1; }

    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }
    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); fclose(in); return 1; }

    char line[16384];
    if (!fgets(line, sizeof(line), in)) {
        fprintf(stderr, "empty input\n");
        fclose(in);
        fclose(out);
        return 1;
    }
    trim_line_end(line);
    fprintf(out,
        "%s,best_slime_chunks_17x17,best_slime_window_chunk_x,best_slime_window_chunk_z,"
        "best_slime_window_min_block_x,best_slime_window_min_block_z,"
        "best_slime_window_max_block_x,best_slime_window_max_block_z,"
        "slime_window_chunks,slime_search_radius_blocks,min_slime_chunks,slime_backend\n",
        line);

    int64_t processed = 0;
    int64_t hits = 0;
    while (fgets(line, sizeof(line), in)) {
        if (limit > 0 && processed >= limit) break;
        trim_line_end(line);
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        int64_t seed = parse_seed_from_csv_line(line);
        BestWindow best = find_best_slime_window_cuda(
            seed, chunk_min, width, window_count, window_chunks, threads, blocks,
            d_grid, d_vertical, d_block_bests, h_block_bests);
        if (best.best >= min_slime_chunks) {
            int best_x = chunk_min + best.wx;
            int best_z = chunk_min + best.wz;
            int min_x = best_x * 16;
            int min_z = best_z * 16;
            int max_x = (best_x + window_chunks) * 16 - 1;
            int max_z = (best_z + window_chunks) * 16 - 1;
            fprintf(out, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,cuda\n",
                line, best.best, best_x, best_z, min_x, min_z, max_x, max_z,
                window_chunks, radius_blocks, min_slime_chunks);
            hits++;
        }
        processed++;
        fprintf(stderr, "processed=%" PRId64 " hits=%" PRId64 " seed=%" PRId64 " best_slime=%d backend=cuda\n",
            processed, hits, seed, best.best);
        fflush(stderr);
    }

    fclose(in);
    fclose(out);
    free(h_block_bests);
    cudaFree(d_grid);
    cudaFree(d_vertical);
    cudaFree(d_block_bests);
    fprintf(stderr, "done: processed=%" PRId64 " hits=%" PRId64 " output=%s backend=cuda\n", processed, hits, out_path);
    return 0;
}
