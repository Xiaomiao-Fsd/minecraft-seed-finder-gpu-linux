// CUDA whole-world / radius Witch Hut (Swamp Hut) lowest-Y postprocessor.
//
// Reads a CSV whose first column is a Minecraft Java seed, scans Swamp Hut
// structure regions on the GPU, applies Java 26.2 biome-noise viability for
// swamp huts, computes cubiomes-compatible approximate surface Y from the 1.18+
// depth parameter, and writes one lowest-Y hut per seed sorted by Y ascending.
//
// Version note: the Swamp Hut structure-set constants used here are the modern
// Java constants used by cubiomes for MC_1_13+ / MC_NEWEST and expected for
// Java 26.2 unless Mojang changes the structure set report:
//   salt=14357620, spacing=32 chunks, separation=8 chunks => chunkRange=24.
// Biome matching uses minecraft_26_2_biome_params_cuda.h generated from the
// official Java 26.2 data-generator biome parameter report.
//
// Build:
//   nvcc -O3 -std=c++17 -o witch_hut_y_filter_cuda witch_hut_y_filter_cuda.cu

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "biome_noise_cuda.h"

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;
static const int SWAMP_HUT_SALT = 14357620;
static const int SWAMP_HUT_REGION_SIZE = 32;
static const int SWAMP_HUT_CHUNK_RANGE = 24;
static const int DEFAULT_RADIUS_BLOCKS = 60000;
static const int MINECRAFT_WORLD_BORDER_BLOCKS = 29999984;

struct BestHut {
    int found;
    int64_t seed;
    int x;
    int z;
    int chunk_x;
    int chunk_z;
    int region_x;
    int region_z;
    int sample_x;
    int sample_z;
    float y;
    int y_block;
    int biome_id;
    unsigned long long checked;
    unsigned long long viable;
};

__device__ __host__ static inline int floor_div_i(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r && ((r > 0) != (b > 0))) q--;
    return q;
}

__device__ __forceinline__ uint64_t next_seed(uint64_t seed) {
    return (seed * JAVA_MULT + JAVA_ADD) & JAVA_MASK;
}

__device__ __forceinline__ int next31_mod(uint64_t *seed, int bound) {
    *seed = next_seed(*seed);
    return (int)(*seed >> 17) % bound;
}

__device__ __forceinline__ void swamp_hut_pos(uint64_t world_seed, int rx, int rz, int *out_x, int *out_z) {
    uint64_t s = world_seed +
        (uint64_t)((int64_t)rx * 341873128712LL) +
        (uint64_t)((int64_t)rz * 132897987541LL) +
        (uint64_t)SWAMP_HUT_SALT;
    s = (s ^ JAVA_MULT) & JAVA_MASK;
    int x = next31_mod(&s, SWAMP_HUT_CHUNK_RANGE);
    int z = next31_mod(&s, SWAMP_HUT_CHUNK_RANGE);
    *out_x = (int)(((int64_t)rx * SWAMP_HUT_REGION_SIZE + x) << 4);
    *out_z = (int)(((int64_t)rz * SWAMP_HUT_REGION_SIZE + z) << 4);
}

__device__ __forceinline__ bool is_swamp_hut_biome_26_2_cuda(int id) {
    // Vanilla Swamp Hut / Witch Hut structure biome tag is swamp, not mangrove swamp.
    return id == BIOME_SWAMP;
}

__device__ int sample_biome_and_approx_y_26_2_cuda(
    const BiomeNoiseCuda *bn,
    int x,
    int y_quart,
    int z,
    int *biome_id,
    float *approx_surface_y
) {
    float t = 0, h = 0, c = 0, e = 0, d = 0, w = 0;
    double px = x, pz = z;
    px += sample_double_perlin_cuda(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
    pz += sample_double_perlin_cuda(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;
    c = (float)sample_double_perlin_cuda(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    e = (float)sample_double_perlin_cuda(&bn->climate[NP_EROSION], px, 0, pz);
    w = (float)sample_double_perlin_cuda(&bn->climate[NP_WEIRDNESS], px, 0, pz);
    float np_param[] = { c, e, -3.0F * (fabsf(fabsf(w) - 0.6666667F) - 0.33333334F), w };
    float off = get_spline_cuda(bn->sp, np_param) + 0.015F;
    d = 1.0F - (y_quart * 4) / 128.0F - 83.0F / 160.0F + off;
    t = (float)sample_double_perlin_cuda(&bn->climate[NP_TEMPERATURE], px, 0, pz);
    h = (float)sample_double_perlin_cuda(&bn->climate[NP_HUMIDITY], px, 0, pz);

    int64_t depth_param = (int64_t)(10000.0F * d);
    uint64_t np[6];
    np[0] = (uint64_t)((int64_t)(10000.0F * t));
    np[1] = (uint64_t)((int64_t)(10000.0F * h));
    np[2] = (uint64_t)((int64_t)(10000.0F * c));
    np[3] = (uint64_t)((int64_t)(10000.0F * e));
    np[4] = (uint64_t)depth_param;
    np[5] = (uint64_t)((int64_t)(10000.0F * w));
    int id = climate_to_biome_26_2_cuda(np);
    if (biome_id) *biome_id = id;
    if (approx_surface_y) *approx_surface_y = (float)depth_param / 76.0f;
    return id;
}

__device__ __forceinline__ bool better_hut(const BestHut &a, const BestHut &b) {
    if (!b.found) return true;
    if (!a.found) return false;
    if (a.y < b.y) return true;
    if (a.y > b.y) return false;
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return a.z < b.z;
}

__global__ void init_one_biome_noise_kernel(BiomeNoiseCuda *bn, uint64_t seed) {
    init_biome_noise_cuda(bn);
    set_biome_seed_cuda(bn, seed);
}

__global__ void scan_swamp_hut_regions_kernel(
    const BiomeNoiseCuda *bn,
    int64_t signed_seed,
    int reg_min,
    int reg_span,
    int64_t batch_start,
    int64_t batch_count,
    int radius,
    bool circle,
    bool require_biome,
    bool negative_y_only,
    bool max_y_enabled,
    float max_y,
    BestHut *block_bests,
    unsigned long long *counts
) {
    extern __shared__ BestHut shared_best[];
    __shared__ unsigned int shared_checked;
    __shared__ unsigned int shared_viable;
    if (threadIdx.x == 0) {
        shared_checked = 0;
        shared_viable = 0;
    }
    __syncthreads();

    BestHut local = {};
    local.seed = signed_seed;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_count) {
        int64_t region_index = batch_start + tid;
        int rz = reg_min + (int)(region_index / reg_span);
        int rx = reg_min + (int)(region_index - (int64_t)(rz - reg_min) * reg_span);

        int x = 0, z = 0;
        swamp_hut_pos((uint64_t)signed_seed, rx, rz, &x, &z);
        bool in_bounds = x >= -radius && x <= radius && z >= -radius && z <= radius;
        if (in_bounds && circle) {
            int64_t xx = (int64_t)x;
            int64_t zz = (int64_t)z;
            in_bounds = xx * xx + zz * zz <= (int64_t)radius * (int64_t)radius;
        }

        if (in_bounds) {
            atomicAdd(&shared_checked, 1U);
            int chunk_x = x >> 4;
            int chunk_z = z >> 4;
            int qx = chunk_x * 4 + 2;
            int qz = chunk_z * 4 + 2;

            int viability_id = -1;
            sample_biome_and_approx_y_26_2_cuda(bn, qx, 319 >> 2, qz, &viability_id, NULL);
            if (!require_biome || is_swamp_hut_biome_26_2_cuda(viability_id)) {
                atomicAdd(&shared_viable, 1U);
                int surface_biome_id = -1;
                float approx_y = 0.0f;
                sample_biome_and_approx_y_26_2_cuda(bn, qx, 0, qz, &surface_biome_id, &approx_y);
                if ((!negative_y_only || approx_y < 0.0f) && (!max_y_enabled || approx_y <= max_y)) {
                    local.found = 1;
                    local.x = x;
                    local.z = z;
                    local.chunk_x = chunk_x;
                    local.chunk_z = chunk_z;
                    local.region_x = rx;
                    local.region_z = rz;
                    local.sample_x = chunk_x * 16 + 8;
                    local.sample_z = chunk_z * 16 + 8;
                    local.y = approx_y;
                    local.y_block = (int)floorf(approx_y + 0.5f);
                    local.biome_id = surface_biome_id;
                }
            }
        }
    }

    shared_best[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            BestHut other = shared_best[threadIdx.x + stride];
            if (better_hut(other, shared_best[threadIdx.x])) shared_best[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_bests[blockIdx.x] = shared_best[0];
        atomicAdd(&counts[0], (unsigned long long)shared_checked);
        atomicAdd(&counts[1], (unsigned long long)shared_viable);
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

static double parse_double_arg(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !end || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static bool parse_seed_line(const char *line, int64_t *seed) {
    const char *p = line;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '-' && !isdigit((unsigned char)*p)) return false;
    errno = 0;
    char *end = NULL;
    long long v = strtoll(p, &end, 10);
    if (errno || end == p) return false;
    if (*end && *end != ',' && *end != '\n' && *end != '\r' && !isspace((unsigned char)*end)) return false;
    *seed = (int64_t)v;
    return true;
}

static bool host_better_hut(const BestHut &a, const BestHut &b) {
    if (!b.found) return true;
    if (!a.found) return false;
    if (a.y < b.y) return true;
    if (a.y > b.y) return false;
    if (a.seed < b.seed) return true;
    if (a.seed > b.seed) return false;
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return a.z < b.z;
}

static bool sort_huts(const BestHut &a, const BestHut &b) {
    if (a.found != b.found) return a.found > b.found;
    if (!a.found) return a.seed < b.seed;
    if (a.y != b.y) return a.y < b.y;
    if (a.seed != b.seed) return a.seed < b.seed;
    if (a.x != b.x) return a.x < b.x;
    return a.z < b.z;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out witch_huts_lowest.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N    search square +/-N blocks from origin (default 60000)\n"
        "  --whole-world        search full Minecraft world-border square +/-29999984\n"
        "  --circle             use Euclidean radius instead of square bounds\n"
        "  --negative-y-only    only keep huts with hut_y_approx < 0\n"
        "  --max-y N            only keep huts with hut_y_approx <= N\n"
        "  --no-biome-check     include structure attempts without swamp biome viability\n"
        "  --include-missing    output seeds with no hut matching filters, sorted after hits\n"
        "  --limit N            process at most N input seed rows, 0 = all (default 0)\n"
        "  --top N              write at most N found rows after sorting, 0 = all (default 0)\n"
        "  --batch-regions N    structure regions per GPU launch (default 8388608)\n"
        "  --threads N          CUDA threads per block, power of two (default 256)\n"
        "  --progress N         print progress every N seeds, 0 = silent (default 1)\n"
        "  --progress-batches N print batch progress every N GPU launches, 0 = silent (default 100)\n"
        "  --mc 26.2            accepted; biome parameters are official Java 26.2\n",
        argv0);
}

static BestHut scan_seed_cuda(
    int64_t seed,
    int radius,
    bool whole_world,
    bool circle,
    bool require_biome,
    bool negative_y_only,
    bool max_y_enabled,
    float max_y,
    int64_t batch_regions,
    int threads,
    long progress_batches
) {
    BestHut best;
    memset(&best, 0, sizeof(best));
    best.seed = seed;

    int region_blocks = SWAMP_HUT_REGION_SIZE * 16;
    int reg_min = floor_div_i(-radius, region_blocks) - 1;
    int reg_max = floor_div_i(radius, region_blocks) + 1;
    int reg_span = reg_max - reg_min + 1;
    int64_t total_regions = (int64_t)reg_span * (int64_t)reg_span;

    if (whole_world) {
        fprintf(stderr,
            "seed=%" PRId64 ": --whole-world scans +/- %d blocks; Swamp_Hut regions=%d x %d = %" PRId64 "\n",
            seed, radius, reg_span, reg_span, total_regions);
    }

    BiomeNoiseCuda *d_bn = NULL;
    BestHut *d_block_bests = NULL;
    unsigned long long *d_counts = NULL;
    cuda_check(cudaMalloc(&d_bn, sizeof(BiomeNoiseCuda)), "cudaMalloc d_bn");
    init_one_biome_noise_kernel<<<1, 1>>>(d_bn, (uint64_t)seed);
    cuda_check(cudaGetLastError(), "init_one_biome_noise_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "init_one_biome_noise_kernel sync");

    int max_blocks = (int)((batch_regions + threads - 1) / threads);
    cuda_check(cudaMalloc(&d_block_bests, (size_t)max_blocks * sizeof(BestHut)), "cudaMalloc d_block_bests");
    cuda_check(cudaMalloc(&d_counts, 2 * sizeof(unsigned long long)), "cudaMalloc d_counts");
    std::vector<BestHut> h_block_bests((size_t)max_blocks);

    unsigned long long total_checked = 0;
    unsigned long long total_viable = 0;
    int64_t batch_index = 0;
    for (int64_t start = 0; start < total_regions; start += batch_regions, batch_index++) {
        int64_t count = std::min(batch_regions, total_regions - start);
        int blocks = (int)((count + threads - 1) / threads);
        cuda_check(cudaMemset(d_counts, 0, 2 * sizeof(unsigned long long)), "cudaMemset d_counts");
        scan_swamp_hut_regions_kernel<<<blocks, threads, (size_t)threads * sizeof(BestHut)>>>(
            d_bn, seed, reg_min, reg_span, start, count, radius, circle, require_biome,
            negative_y_only, max_y_enabled, max_y, d_block_bests, d_counts);
        cuda_check(cudaGetLastError(), "scan_swamp_hut_regions_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "scan_swamp_hut_regions_kernel sync");
        cuda_check(cudaMemcpy(h_block_bests.data(), d_block_bests, (size_t)blocks * sizeof(BestHut), cudaMemcpyDeviceToHost), "copy block bests");
        unsigned long long h_counts[2] = {0, 0};
        cuda_check(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost), "copy counts");
        total_checked += h_counts[0];
        total_viable += h_counts[1];
        for (int i = 0; i < blocks; i++) {
            if (host_better_hut(h_block_bests[(size_t)i], best)) best = h_block_bests[(size_t)i];
        }
        best.seed = seed;
        if (progress_batches > 0 && (batch_index + 1) % progress_batches == 0) {
            double pct = 100.0 * (double)(start + count) / (double)total_regions;
            fprintf(stderr, "seed=%" PRId64 " batch=%" PRId64 " progress=%.2f%% checked=%llu viable=%llu best=%s\n",
                    seed, batch_index + 1, pct, total_checked, total_viable, best.found ? "yes" : "no");
            fflush(stderr);
        }
    }

    best.checked = total_checked;
    best.viable = total_viable;
    best.seed = seed;

    cudaFree(d_counts);
    cudaFree(d_block_bests);
    cudaFree(d_bn);
    return best;
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    int radius = DEFAULT_RADIUS_BLOCKS;
    bool whole_world = false;
    bool circle = false;
    bool require_biome = true;
    bool negative_y_only = false;
    bool max_y_enabled = false;
    float max_y = 0.0f;
    bool include_missing = false;
    long limit = 0;
    long top = 0;
    long progress = 1;
    long progress_batches = 100;
    int64_t batch_regions = 8388608LL;
    int threads = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) { radius = (int)parse_ll_arg(argv[++i], "radius-blocks"); whole_world = false; }
        else if (!strcmp(argv[i], "--whole-world")) { radius = MINECRAFT_WORLD_BORDER_BLOCKS; whole_world = true; }
        else if (!strcmp(argv[i], "--circle")) circle = true;
        else if (!strcmp(argv[i], "--negative-y-only")) negative_y_only = true;
        else if (!strcmp(argv[i], "--max-y") && i + 1 < argc) { max_y = (float)parse_double_arg(argv[++i], "max-y"); max_y_enabled = true; }
        else if (!strcmp(argv[i], "--no-biome-check")) require_biome = false;
        else if (!strcmp(argv[i], "--include-missing")) include_missing = true;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = (long)parse_ll_arg(argv[++i], "limit");
        else if (!strcmp(argv[i], "--top") && i + 1 < argc) top = (long)parse_ll_arg(argv[++i], "top");
        else if (!strcmp(argv[i], "--batch-regions") && i + 1 < argc) batch_regions = parse_ll_arg(argv[++i], "batch-regions");
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (int)parse_ll_arg(argv[++i], "threads");
        else if (!strcmp(argv[i], "--progress") && i + 1 < argc) progress = (long)parse_ll_arg(argv[++i], "progress");
        else if (!strcmp(argv[i], "--progress-batches") && i + 1 < argc) progress_batches = (long)parse_ll_arg(argv[++i], "progress-batches");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) {
            const char *mc = argv[++i];
            if (strcmp(mc, "26.2") && strcmp(mc, "newest")) {
                fprintf(stderr, "WARNING: CUDA witch hut helper uses Java 26.2 biome parameters; treating --mc %s as 26.2.\n", mc);
            }
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 2; }
    }

    if (!in_path || !out_path || radius <= 0 || limit < 0 || top < 0 || batch_regions <= 0 || threads <= 0 || progress < 0 || progress_batches < 0) {
        usage(argv[0]);
        return 2;
    }
    if ((threads & (threads - 1)) != 0) {
        fprintf(stderr, "--threads must be a power of two for shared-memory reduction.\n");
        return 2;
    }

    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }
    std::vector<int64_t> seeds;
    char line[16384];
    long skipped = 0;
    while (fgets(line, sizeof(line), in)) {
        int64_t seed = 0;
        if (!parse_seed_line(line, &seed)) { skipped++; continue; }
        if (limit > 0 && (long)seeds.size() >= limit) break;
        seeds.push_back(seed);
    }
    fclose(in);

    std::vector<BestHut> results;
    results.reserve(seeds.size());
    long hits = 0;
    for (size_t i = 0; i < seeds.size(); i++) {
        BestHut r = scan_seed_cuda(seeds[i], radius, whole_world, circle, require_biome,
                                   negative_y_only, max_y_enabled, max_y, batch_regions,
                                   threads, progress_batches);
        if (r.found) { hits++; results.push_back(r); }
        else if (include_missing) { results.push_back(r); }
        if (progress > 0 && ((long)i + 1) % progress == 0) {
            fprintf(stderr, "processed=%zu/%zu hits=%ld skipped_non_seed_lines=%ld\n", i + 1, seeds.size(), hits, skipped);
            fflush(stderr);
        }
    }

    std::sort(results.begin(), results.end(), sort_huts);

    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); return 1; }
    fprintf(out,
        "seed,hut_x,hut_y_approx,hut_y_approx_block,hut_z,"
        "hut_chunk_x,hut_chunk_z,hut_region_x,hut_region_z,"
        "height_sample_x,height_sample_z,surface_biome_id,"
        "checked_structure_positions,viable_huts_in_scope,"
        "scope_blocks,mc_approx,backend,method,note\n");

    const char *missing_note = negative_y_only
        ? (whole_world ? "no negative-Y hut found in world border" : "no negative-Y hut found in radius")
        : (max_y_enabled
            ? (whole_world ? "no hut at or below max-y found in world border" : "no hut at or below max-y found in radius")
            : (whole_world ? "no hut found in world border" : "no hut found in radius"));

    long written = 0;
    for (size_t i = 0; i < results.size(); i++) {
        const BestHut &r = results[i];
        if (top > 0 && r.found && written >= top) break;
        if (!r.found) {
            fprintf(out, "%" PRId64 ",,,,,,,,,,,,%llu,%llu,%d,26.2,cuda,missing,%s\n",
                    r.seed, r.checked, r.viable, radius, missing_note);
            continue;
        }
        fprintf(out,
            "%" PRId64 ",%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%d,26.2,cuda,"
            "cuda_structure_scan_plus_26_2_biome_noise_depth,"
            "approximate Y from 1.18+ depth parameter; validate in-game for exact block-level terrain\n",
            r.seed, r.x, r.y, r.y_block, r.z,
            r.chunk_x, r.chunk_z, r.region_x, r.region_z,
            r.sample_x, r.sample_z, r.biome_id,
            r.checked, r.viable, radius);
        written++;
    }
    fclose(out);

    fprintf(stderr, "done: processed=%zu hits=%ld skipped_non_seed_lines=%ld wrote=%ld output=%s\n",
            seeds.size(), hits, skipped, written, out_path);
    if (!results.empty() && results[0].found) {
        fprintf(stderr, "best: seed=%" PRId64 " hut=(%d,%d,%d) approx_y=%.3f\n",
                results[0].seed, results[0].x, results[0].y_block, results[0].z, results[0].y);
    }
    return 0;
}
