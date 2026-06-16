// CUDA all-overworld-biomes filter for candidate Minecraft Java seeds.
//
// CPU work is limited to CSV parsing and writing. Biome-noise initialization,
// grid sampling, biome de-duplication, and required-mask checks run on CUDA.
//
// The biome implementation uses the official Minecraft Java 26.2
// data-generator biome parameter report.

#include <cuda_runtime.h>

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "biome_noise_cuda.h"

__global__ void init_biome_noise_kernel(BiomeNoiseCuda *bn, const int64_t *seeds, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_biome_noise_cuda(&bn[idx]);
    set_biome_seed_cuda(&bn[idx], (uint64_t)seeds[idx]);
}

__global__ void scan_all_biomes_kernel(
    const BiomeNoiseCuda *bn,
    int count,
    int start_x,
    int start_z,
    int sx,
    int sz,
    int scale,
    uint64_t req0,
    uint64_t req1,
    uint64_t req2,
    uint64_t req3,
    uint64_t *seen,
    int *ok
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= count) return;

    if (threadIdx.x < 4) seen[seed_idx * 4 + threadIdx.x] = 0;
    if (threadIdx.x == 0) ok[seed_idx] = 0;
    __syncthreads();

    int total = sx * sz;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int x = idx % sx;
        int z = idx / sx;
        int id = sample_biome_scaled_cuda(&bn[seed_idx], scale, start_x + x, 0, start_z + z);
        if ((unsigned int)id < 256U) {
            atomicOr(
                (unsigned long long *)&seen[seed_idx * 4 + (id >> 6)],
                (unsigned long long)(1ULL << (id & 63)));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t s0 = seen[seed_idx * 4 + 0];
        uint64_t s1 = seen[seed_idx * 4 + 1];
        uint64_t s2 = seen[seed_idx * 4 + 2];
        uint64_t s3 = seen[seed_idx * 4 + 3];
        ok[seed_idx] = ((s0 & req0) == req0) &&
                       ((s1 & req1) == req1) &&
                       ((s2 & req2) == req2) &&
                       ((s3 & req3) == req3);
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

static void trim_line_end(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out all_biomes.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N    square +/-N from origin (default 10000)\n"
        "  --scale N            biome range scale: 4,16,64,256 (default 64)\n"
        "  --mc 26.2  accepted, uses official 26.2 biome parameters\n"
        "  --limit N            process at most N candidate rows, 0 = all\n"
        "  --threads N          CUDA threads per seed block (default 256)\n"
        "  --batch-size N       candidate seeds per GPU batch (default 512)\n",
        argv0);
}

static void process_batch(
    FILE *out,
    const std::vector<std::string> &rows,
    const std::vector<int64_t> &seeds,
    int radius,
    int scale,
    int threads,
    int required_count,
    const uint64_t required[4],
    BiomeNoiseCuda *d_bn,
    int64_t *d_seeds,
    uint64_t *d_seen,
    int *d_ok,
    std::vector<uint64_t> &h_seen,
    std::vector<int> &h_ok,
    int64_t *hits
) {
    int n = (int)seeds.size();
    if (n == 0) return;
    int start_x = -radius / scale;
    int start_z = -radius / scale;
    int sx = (2 * radius) / scale + 1;
    int sz = sx;

    cuda_check(cudaMemcpy(d_seeds, seeds.data(), (size_t)n * sizeof(int64_t), cudaMemcpyHostToDevice), "copy seeds");
    int init_blocks = (n + 127) / 128;
    init_biome_noise_kernel<<<init_blocks, 128>>>(d_bn, d_seeds, n);
    cuda_check(cudaGetLastError(), "init_biome_noise_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "init_biome_noise_kernel sync");

    scan_all_biomes_kernel<<<n, threads>>>(
        d_bn, n, start_x, start_z, sx, sz, scale,
        required[0], required[1], required[2], required[3],
        d_seen, d_ok);
    cuda_check(cudaGetLastError(), "scan_all_biomes_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "scan_all_biomes_kernel sync");

    h_seen.resize((size_t)n * 4);
    h_ok.resize((size_t)n);
    cuda_check(cudaMemcpy(h_seen.data(), d_seen, (size_t)n * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy seen masks");
    cuda_check(cudaMemcpy(h_ok.data(), d_ok, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost), "copy ok flags");

    for (int i = 0; i < n; i++) {
        if (!h_ok[i]) continue;
        fprintf(out,
            "%s,%d,1,26.2,%d,%d,cuda_biome_noise_26_2,0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 "\n",
            rows[i].c_str(),
            required_count,
            radius,
            scale,
            h_seen[(size_t)i * 4 + 0],
            h_seen[(size_t)i * 4 + 1],
            h_seen[(size_t)i * 4 + 2],
            h_seen[(size_t)i * 4 + 3]);
        (*hits)++;
    }
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    int radius = 10000;
    int scale = 64;
    int threads = 256;
    int batch_size = 512;
    int64_t limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_ll_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = (int)parse_ll_arg(argv[++i], "scale");
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_ll_arg(argv[++i], "limit");
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (int)parse_ll_arg(argv[++i], "threads");
        else if (!strcmp(argv[i], "--batch-size") && i + 1 < argc) batch_size = (int)parse_ll_arg(argv[++i], "batch-size");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) ++i;
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path) { usage(argv[0]); return 2; }
    if (radius <= 0 || scale <= 0 || threads <= 0 || batch_size <= 0) {
        fprintf(stderr, "radius, scale, threads, and batch-size must be positive.\n");
        return 2;
    }
    if (scale != 4 && scale != 16 && scale != 64 && scale != 256) {
        fprintf(stderr, "scale must be one of 4,16,64,256 for the CUDA biome helper.\n");
        return 2;
    }

    uint64_t required[4];
    int required_count = 0;
    required_overworld_biomes_26_2(required, &required_count);
    fprintf(stderr,
        "required_overworld_biomes=%d mc_ref=26.2 scale=%d radius=%d backend=cuda_biome_noise_26_2\n",
        required_count, scale, radius);

    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }
    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); fclose(in); return 1; }

    BiomeNoiseCuda *d_bn = NULL;
    int64_t *d_seeds = NULL;
    uint64_t *d_seen = NULL;
    int *d_ok = NULL;
    cuda_check(cudaMalloc((void **)&d_bn, (size_t)batch_size * sizeof(BiomeNoiseCuda)), "cudaMalloc biome noise");
    cuda_check(cudaMalloc((void **)&d_seeds, (size_t)batch_size * sizeof(int64_t)), "cudaMalloc seeds");
    cuda_check(cudaMalloc((void **)&d_seen, (size_t)batch_size * 4 * sizeof(uint64_t)), "cudaMalloc seen masks");
    cuda_check(cudaMalloc((void **)&d_ok, (size_t)batch_size * sizeof(int)), "cudaMalloc ok flags");

    char line[16384];
    if (!fgets(line, sizeof(line), in)) {
        fprintf(stderr, "empty input\n");
        fclose(in);
        fclose(out);
        return 1;
    }
    trim_line_end(line);
    fprintf(out,
        "%s,required_biome_count,check_return,all_biomes_mc_ref,all_biomes_radius_blocks,biome_scale,"
        "all_biomes_backend,seen_mask0,seen_mask1,seen_mask2,seen_mask3\n",
        line);

    std::vector<std::string> rows;
    std::vector<int64_t> seeds;
    std::vector<uint64_t> h_seen;
    std::vector<int> h_ok;
    rows.reserve((size_t)batch_size);
    seeds.reserve((size_t)batch_size);

    int64_t processed = 0;
    int64_t hits = 0;
    while (fgets(line, sizeof(line), in)) {
        if (limit > 0 && processed >= limit) break;
        trim_line_end(line);
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        rows.emplace_back(line);
        seeds.push_back(parse_seed_from_csv_line(line));
        processed++;
        if ((int)rows.size() >= batch_size) {
            process_batch(out, rows, seeds, radius, scale, threads, required_count, required,
                d_bn, d_seeds, d_seen, d_ok, h_seen, h_ok, &hits);
            fprintf(stderr, "processed=%" PRId64 " hits=%" PRId64 " backend=cuda_biome_noise_26_2\n", processed, hits);
            fflush(stderr);
            rows.clear();
            seeds.clear();
        }
    }
    process_batch(out, rows, seeds, radius, scale, threads, required_count, required,
        d_bn, d_seeds, d_seen, d_ok, h_seen, h_ok, &hits);

    fclose(in);
    fclose(out);
    cudaFree(d_bn);
    cudaFree(d_seeds);
    cudaFree(d_seen);
    cudaFree(d_ok);
    fprintf(stderr, "done: processed=%" PRId64 " hits=%" PRId64 " output=%s backend=cuda_biome_noise_26_2\n",
        processed, hits, out_path);
    return 0;
}
