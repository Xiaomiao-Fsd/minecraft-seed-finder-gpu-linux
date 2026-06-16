// CUDA coarse prefilter for Minecraft Java slime-chunk density.
// Build: nvcc -O3 -std=c++17 -o slime_prefilter_cuda slime_prefilter_cuda.cu
// One CUDA thread scans one world seed across sampled candidate circle centers.

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

struct Offset { int dx; int dz; };
struct Candidate { int64_t seed; int best; int best_cx; int best_cz; };

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;

__device__ __forceinline__ uint64_t splitmix64_dev(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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
    // Match vanilla Java exactly: several terms are 32-bit int operations
    // before sign-extension/promotion to long.
    s += (uint64_t)java_int_to_long_dev(java_int_mul3_dev(cx, cx, 4987142));
    s += (uint64_t)java_int_to_long_dev(java_int_mul2_dev(cx, 5947611));
    s += (uint64_t)(java_int_to_long_dev(java_int_mul2_dev(cz, cz)) * 4392871LL);
    s += (uint64_t)java_int_to_long_dev(java_int_mul2_dev(cz, 389711));
    s ^= 987234911ULL;
    uint64_t rnd = (s ^ JAVA_MULT) & JAVA_MASK;
    return java_next_int_dev(&rnd, 10) == 0;
}

__global__ void slime_prefilter_kernel(
    int64_t start,
    int64_t count,
    int threshold,
    int search_radius_chunks,
    int center_samples,
    const Offset *offsets,
    int noff,
    Candidate *candidates,
    unsigned int *candidate_count,
    unsigned int max_output
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int64_t seed = start + idx;
    int best = -1;
    int best_cx = 0;
    int best_cz = 0;
    int total_centers = center_samples + 1;
    uint64_t sm = ((uint64_t)seed) ^ 0xD1B54A32D192ED03ULL;

    for (int sample = 0; sample < total_centers; sample++) {
        int cx = 0;
        int cz = 0;
        if (sample > 0) {
            int span = 2 * search_radius_chunks + 1;
            do {
                cx = (int)(splitmix64_dev(&sm) % (uint64_t)span) - search_radius_chunks;
                cz = (int)(splitmix64_dev(&sm) % (uint64_t)span) - search_radius_chunks;
            } while (cx * cx + cz * cz > search_radius_chunks * search_radius_chunks);
        }

        int cnt = 0;
        for (int i = 0; i < noff; i++) {
            if (is_slime_chunk_dev(seed, cx + offsets[i].dx, cz + offsets[i].dz)) cnt++;
        }
        if (cnt > best) {
            best = cnt;
            best_cx = cx;
            best_cz = cz;
            if (best >= threshold) break;
        }
    }

    if (best >= threshold) {
        unsigned int slot = atomicAdd(candidate_count, 1U);
        if (slot < max_output) {
            candidates[slot].seed = seed;
            candidates[slot].best = best;
            candidates[slot].best_cx = best_cx;
            candidates[slot].best_cz = best_cz;
        }
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --start N --count N --out FILE [options]\n"
        "Options:\n"
        "  --threshold N             candidate if best count >= N (default 28)\n"
        "  --circle-radius-chunks N  radius for slime circle in chunks (default 8 = 128 blocks)\n"
        "  --search-radius-chunks N  sample centers within +/-N chunks (default 625 = 10000 blocks)\n"
        "  --center-samples N        sampled centers per seed, plus origin (default 64)\n"
        "  --max-candidates N        stop after N candidates, 0 = no cap (default 0)\n"
        "  --append                  append to output CSV instead of overwriting\n"
        "  --batch-size N            seeds per GPU launch (default 1048576)\n"
        "  --threads N               CUDA threads per block (default 256)\n",
        argv0);
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

static void cuda_check(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static int cmp_candidate_seed(const void *a, const void *b) {
    const Candidate *ca = (const Candidate *)a;
    const Candidate *cb = (const Candidate *)b;
    if (ca->seed < cb->seed) return -1;
    if (ca->seed > cb->seed) return 1;
    return 0;
}

int main(int argc, char **argv) {
    int64_t start = 0;
    int64_t count = -1;
    int threshold = 28;
    int circle_radius_chunks = 8;
    int search_radius_chunks = 625;
    int center_samples = 64;
    long max_candidates = 0;
    long long batch_size = 1048576LL;
    int threads = 256;
    const char *out_path = NULL;
    bool append = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_ll_arg(argv[++i], "start");
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = parse_ll_arg(argv[++i], "count");
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = (int)parse_ll_arg(argv[++i], "threshold");
        else if (!strcmp(argv[i], "--circle-radius-chunks") && i + 1 < argc) circle_radius_chunks = (int)parse_ll_arg(argv[++i], "circle-radius-chunks");
        else if (!strcmp(argv[i], "--search-radius-chunks") && i + 1 < argc) search_radius_chunks = (int)parse_ll_arg(argv[++i], "search-radius-chunks");
        else if (!strcmp(argv[i], "--center-samples") && i + 1 < argc) center_samples = (int)parse_ll_arg(argv[++i], "center-samples");
        else if (!strcmp(argv[i], "--max-candidates") && i + 1 < argc) max_candidates = (long)parse_ll_arg(argv[++i], "max-candidates");
        else if (!strcmp(argv[i], "--batch-size") && i + 1 < argc) batch_size = parse_ll_arg(argv[++i], "batch-size");
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = (int)parse_ll_arg(argv[++i], "threads");
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--append")) append = true;
        else { usage(argv[0]); return 2; }
    }
    if (count < 0 || !out_path) { usage(argv[0]); return 2; }
    if (circle_radius_chunks < 0 || search_radius_chunks < 0 || center_samples < 0 || batch_size <= 0 || threads <= 0) {
        fprintf(stderr, "Radii, sample counts, batch size and threads must be positive/non-negative.\n");
        return 2;
    }

    int max_offsets = (2 * circle_radius_chunks + 1) * (2 * circle_radius_chunks + 1);
    Offset *h_offsets = (Offset *)calloc((size_t)max_offsets, sizeof(Offset));
    if (!h_offsets) { perror("calloc offsets"); return 1; }
    int noff = 0;
    for (int dz = -circle_radius_chunks; dz <= circle_radius_chunks; dz++) {
        for (int dx = -circle_radius_chunks; dx <= circle_radius_chunks; dx++) {
            if (dx * dx + dz * dz <= circle_radius_chunks * circle_radius_chunks) {
                h_offsets[noff++] = Offset{dx, dz};
            }
        }
    }

    FILE *out = fopen(out_path, append ? "a" : "w");
    if (!out) { perror(out_path); free(h_offsets); return 1; }
    if (!append) {
        fprintf(out, "seed,best_slime_chunks,best_center_chunk_x,best_center_chunk_z,center_samples,circle_radius_chunks,search_radius_chunks\n");
    }

    Offset *d_offsets = NULL;
    cuda_check(cudaMalloc((void **)&d_offsets, (size_t)noff * sizeof(Offset)), "cudaMalloc offsets");
    cuda_check(cudaMemcpy(d_offsets, h_offsets, (size_t)noff * sizeof(Offset), cudaMemcpyHostToDevice), "cudaMemcpy offsets");

    Candidate *d_candidates = NULL;
    unsigned int *d_candidate_count = NULL;
    // Allocate one candidate slot per seed in the batch. This is worst-case safe and simple.
    cuda_check(cudaMalloc((void **)&d_candidates, (size_t)batch_size * sizeof(Candidate)), "cudaMalloc candidates");
    cuda_check(cudaMalloc((void **)&d_candidate_count, sizeof(unsigned int)), "cudaMalloc candidate_count");
    Candidate *h_candidates = (Candidate *)malloc((size_t)batch_size * sizeof(Candidate));
    if (!h_candidates) { perror("malloc candidates"); return 1; }

    long total_candidates = 0;
    int64_t scanned = 0;
    int64_t next_progress = 10000000LL;
    while (scanned < count) {
        int64_t todo64 = count - scanned;
        int64_t batch64 = todo64 < batch_size ? todo64 : batch_size;
        unsigned int batch = (unsigned int)batch64;
        cuda_check(cudaMemset(d_candidate_count, 0, sizeof(unsigned int)), "cudaMemset candidate_count");

        int blocks = (int)((batch64 + threads - 1) / threads);
        slime_prefilter_kernel<<<blocks, threads>>>(
            start + scanned,
            batch64,
            threshold,
            search_radius_chunks,
            center_samples,
            d_offsets,
            noff,
            d_candidates,
            d_candidate_count,
            batch
        );
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "kernel sync");

        unsigned int found = 0;
        cuda_check(cudaMemcpy(&found, d_candidate_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy candidate_count");
        unsigned int capped = found > batch ? batch : found;
        if (capped > 0) {
            cuda_check(cudaMemcpy(h_candidates, d_candidates, (size_t)capped * sizeof(Candidate), cudaMemcpyDeviceToHost), "copy candidates");
            qsort(h_candidates, capped, sizeof(Candidate), cmp_candidate_seed);
            for (unsigned int i = 0; i < capped; i++) {
                if (max_candidates > 0 && total_candidates >= max_candidates) break;
                fprintf(out, "%" PRId64 ",%d,%d,%d,%d,%d,%d\n",
                    h_candidates[i].seed,
                    h_candidates[i].best,
                    h_candidates[i].best_cx,
                    h_candidates[i].best_cz,
                    center_samples,
                    circle_radius_chunks,
                    search_radius_chunks);
                total_candidates++;
            }
        }
        scanned += batch64;
        if (found > batch) {
            fprintf(stderr, "warning: candidate buffer saturated in batch starting at %" PRId64 "; some candidates were dropped\n", start + scanned - batch64);
        }
        if (scanned >= next_progress || scanned >= count) {
            fprintf(stderr, "progress: start=%" PRId64 " scanned=%" PRId64 "/%" PRId64 " candidates=%ld\n",
                start, scanned, count, total_candidates);
            fflush(stderr);
            while (next_progress <= scanned) next_progress += 10000000LL;
        }
        if (max_candidates > 0 && total_candidates >= max_candidates) break;
    }

    fclose(out);
    free(h_offsets);
    free(h_candidates);
    cudaFree(d_offsets);
    cudaFree(d_candidates);
    cudaFree(d_candidate_count);
    fprintf(stderr, "done: scanned=%" PRId64 " candidates=%ld output=%s backend=cuda\n", scanned, total_candidates, out_path);
    return 0;
}
