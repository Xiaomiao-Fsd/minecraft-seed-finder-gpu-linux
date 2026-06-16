// Fast coarse prefilter for Minecraft Java slime-chunk density.
// Build: gcc -O3 -march=native -std=c11 -Wall -Wextra -o slime_prefilter slime_prefilter.c
// This is a deliberately cheap/approximate prefilter: it samples possible circle centers.
// Re-run with larger --center-samples or exact checker before accepting final seeds.

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int dx; int dz; } Offset;

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int java_next_bits(uint64_t *seed, int bits) {
    *seed = ((*seed * JAVA_MULT) + JAVA_ADD) & JAVA_MASK;
    return (int)(*seed >> (48 - bits));
}

static int java_next_int(uint64_t *seed, int bound) {
    int bits, val;
    if ((bound & -bound) == bound) {
        return (int)((bound * (int64_t)java_next_bits(seed, 31)) >> 31);
    }
    do {
        bits = java_next_bits(seed, 31);
        val = bits % bound;
    } while (bits - val + (bound - 1) < 0);
    return val;
}

static int64_t java_int_to_long(uint32_t value) {
    return (value & 0x80000000U) ? (int64_t)value - 0x100000000LL : (int64_t)value;
}

static uint32_t java_int_mul2(int32_t a, int32_t b) {
    return (uint32_t)a * (uint32_t)b;
}

static uint32_t java_int_mul3(int32_t a, int32_t b, int32_t c) {
    return ((uint32_t)a * (uint32_t)b) * (uint32_t)c;
}

static bool is_slime_chunk(int64_t world_seed, int32_t cx, int32_t cz) {
    uint64_t s = (uint64_t)world_seed;
    // Match the vanilla Java expression exactly. The x*x*4987142,
    // x*5947611, z*z and z*389711 sub-expressions are evaluated as
    // 32-bit Java int operations before being cast/promoted to long.
    s += (uint64_t)java_int_to_long(java_int_mul3(cx, cx, 4987142));
    s += (uint64_t)java_int_to_long(java_int_mul2(cx, 5947611));
    s += (uint64_t)(java_int_to_long(java_int_mul2(cz, cz)) * 4392871LL);
    s += (uint64_t)java_int_to_long(java_int_mul2(cz, 389711));
    s ^= 987234911ULL;
    uint64_t rnd = (s ^ JAVA_MULT) & JAVA_MASK;
    return java_next_int(&rnd, 10) == 0;
}

static long parse_long_arg(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
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
        "  --append                  append to output CSV instead of overwriting\n",
        argv0);
}

int main(int argc, char **argv) {
    int64_t start = 0;
    int64_t count = -1;
    int threshold = 28;
    int circle_radius_chunks = 8;
    int search_radius_chunks = 625;
    int center_samples = 64;
    long max_candidates = 0;
    const char *out_path = NULL;
    bool append = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_long_arg(argv[++i], "start");
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = parse_long_arg(argv[++i], "count");
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = (int)parse_long_arg(argv[++i], "threshold");
        else if (!strcmp(argv[i], "--circle-radius-chunks") && i + 1 < argc) circle_radius_chunks = (int)parse_long_arg(argv[++i], "circle-radius-chunks");
        else if (!strcmp(argv[i], "--search-radius-chunks") && i + 1 < argc) search_radius_chunks = (int)parse_long_arg(argv[++i], "search-radius-chunks");
        else if (!strcmp(argv[i], "--center-samples") && i + 1 < argc) center_samples = (int)parse_long_arg(argv[++i], "center-samples");
        else if (!strcmp(argv[i], "--max-candidates") && i + 1 < argc) max_candidates = parse_long_arg(argv[++i], "max-candidates");
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--append")) append = true;
        else { usage(argv[0]); return 2; }
    }
    if (count < 0 || !out_path) { usage(argv[0]); return 2; }
    if (circle_radius_chunks < 0 || search_radius_chunks < 0 || center_samples < 0) {
        fprintf(stderr, "Radii and sample counts must be non-negative.\n");
        return 2;
    }

    int max_offsets = (2 * circle_radius_chunks + 1) * (2 * circle_radius_chunks + 1);
    Offset *offsets = (Offset *)calloc((size_t)max_offsets, sizeof(Offset));
    if (!offsets) { perror("calloc offsets"); return 1; }
    int noff = 0;
    for (int dz = -circle_radius_chunks; dz <= circle_radius_chunks; dz++) {
        for (int dx = -circle_radius_chunks; dx <= circle_radius_chunks; dx++) {
            if (dx * dx + dz * dz <= circle_radius_chunks * circle_radius_chunks) {
                offsets[noff++] = (Offset){dx, dz};
            }
        }
    }

    FILE *out = fopen(out_path, append ? "a" : "w");
    if (!out) { perror(out_path); free(offsets); return 1; }
    if (!append) {
        fprintf(out, "seed,best_slime_chunks,best_center_chunk_x,best_center_chunk_z,center_samples,circle_radius_chunks,search_radius_chunks\n");
    }

    long candidates = 0;
    for (int64_t idx = 0; idx < count; idx++) {
        int64_t seed = start + idx;
        int best = -1;
        int best_cx = 0, best_cz = 0;

        int total_centers = center_samples + 1; // origin plus sampled centers
        uint64_t sm = (uint64_t)seed ^ 0xD1B54A32D192ED03ULL;
        for (int sample = 0; sample < total_centers; sample++) {
            int cx = 0, cz = 0;
            if (sample > 0) {
                int span = 2 * search_radius_chunks + 1;
                // Sample center chunks inside the origin-centered search disk, not the full square.
                // This matches the requirement that the 128-block slime circle is found within
                // 10000 blocks of origin (interpreted as center within radius).
                do {
                    cx = (int)(splitmix64(&sm) % (uint64_t)span) - search_radius_chunks;
                    cz = (int)(splitmix64(&sm) % (uint64_t)span) - search_radius_chunks;
                } while (cx * cx + cz * cz > search_radius_chunks * search_radius_chunks);
            }
            int cnt = 0;
            for (int i = 0; i < noff; i++) {
                if (is_slime_chunk(seed, cx + offsets[i].dx, cz + offsets[i].dz)) cnt++;
            }
            if (cnt > best) {
                best = cnt;
                best_cx = cx;
                best_cz = cz;
                if (best >= threshold) break;
            }
        }

        if (best >= threshold) {
            fprintf(out, "%" PRId64 ",%d,%d,%d,%d,%d,%d\n",
                    seed, best, best_cx, best_cz, center_samples, circle_radius_chunks, search_radius_chunks);
            candidates++;
            if (max_candidates > 0 && candidates >= max_candidates) break;
        }
        if ((idx + 1) % 10000000LL == 0) {
            fprintf(stderr, "progress: start=%" PRId64 " scanned=%" PRId64 "/%" PRId64 " candidates=%ld\n",
                    start, idx + 1, count, candidates);
            fflush(stderr);
        }
    }

    fclose(out);
    free(offsets);
    fprintf(stderr, "done: scanned=%" PRId64 " candidates=%ld output=%s\n", count, candidates, out_path);
    return 0;
}
