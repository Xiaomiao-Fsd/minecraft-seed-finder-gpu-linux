// Exact 17x17 slime-window scanner for candidate Minecraft Java seeds.
//
// This stage reads a CSV whose first column is seed, scans every chunk-aligned
// window that fully fits inside an origin-centered +/-radius block square, and
// appends the maximum slime chunk count found in a window of the configured size.
//
// Build: gcc -O3 -march=native -std=c11 -Wall -Wextra -o slime_window_filter slime_window_filter.c

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t JAVA_MULT = 0x5DEECE66DULL;
static const uint64_t JAVA_ADD = 0xBULL;
static const uint64_t JAVA_MASK = (1ULL << 48) - 1ULL;

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
    s += (uint64_t)java_int_to_long(java_int_mul3(cx, cx, 4987142));
    s += (uint64_t)java_int_to_long(java_int_mul2(cx, 5947611));
    s += (uint64_t)(java_int_to_long(java_int_mul2(cz, cz)) * 4392871LL);
    s += (uint64_t)java_int_to_long(java_int_mul2(cz, 389711));
    s ^= 987234911ULL;
    uint64_t rnd = (s ^ JAVA_MULT) & JAVA_MASK;
    return java_next_int(&rnd, 10) == 0;
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

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out slime_windows.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N     require full window inside square +/-N blocks (default 60000)\n"
        "  --window-chunks N     chunk-aligned square window size (default 17)\n"
        "  --min-slime-chunks N  keep rows with best count >= N (default 0)\n"
        "  --limit N             process at most N candidate rows, 0 = all\n",
        argv0);
}

static void trim_line_end(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

static int find_best_slime_window(
    int64_t seed,
    int radius_blocks,
    int window_chunks,
    int *best_chunk_x,
    int *best_chunk_z
) {
    int64_t window_blocks = (int64_t)window_chunks * 16;
    int64_t start_min64 = ceil_div_i64(-(int64_t)radius_blocks, 16);
    int64_t start_max64 = floor_div_i64((int64_t)radius_blocks - (window_blocks - 1), 16);
    if (start_min64 > start_max64) {
        *best_chunk_x = 0;
        *best_chunk_z = 0;
        return -1;
    }

    int start_min = (int)start_min64;
    int start_max = (int)start_max64;
    int window_count = start_max - start_min + 1;
    int chunk_min = start_min;
    int width = window_count + window_chunks - 1;
    int *col_sums = (int *)calloc((size_t)width, sizeof(int));
    if (!col_sums) {
        perror("calloc col_sums");
        exit(1);
    }

    for (int dz = 0; dz < window_chunks; dz++) {
        int cz = start_min + dz;
        for (int ix = 0; ix < width; ix++) {
            if (is_slime_chunk(seed, chunk_min + ix, cz)) col_sums[ix]++;
        }
    }

    int best = -1;
    int best_x = start_min;
    int best_z = start_min;
    for (int wz_idx = 0; wz_idx < window_count; wz_idx++) {
        int wz = start_min + wz_idx;
        int running = 0;
        for (int ix = 0; ix < window_chunks; ix++) running += col_sums[ix];
        for (int wx_idx = 0; wx_idx < window_count; wx_idx++) {
            int wx = start_min + wx_idx;
            if (running > best) {
                best = running;
                best_x = wx;
                best_z = wz;
            }
            if (wx_idx + window_chunks < width) {
                running += col_sums[wx_idx + window_chunks] - col_sums[wx_idx];
            }
        }

        if (wz_idx + 1 < window_count) {
            int remove_z = wz;
            int add_z = wz + window_chunks;
            for (int ix = 0; ix < width; ix++) {
                if (is_slime_chunk(seed, chunk_min + ix, remove_z)) col_sums[ix]--;
                if (is_slime_chunk(seed, chunk_min + ix, add_z)) col_sums[ix]++;
            }
        }
    }

    free(col_sums);
    *best_chunk_x = best_x;
    *best_chunk_z = best_z;
    return best;
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    int radius_blocks = 60000;
    int window_chunks = 17;
    int min_slime_chunks = 0;
    int64_t limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius_blocks = (int)parse_ll_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--window-chunks") && i + 1 < argc) window_chunks = (int)parse_ll_arg(argv[++i], "window-chunks");
        else if (!strcmp(argv[i], "--min-slime-chunks") && i + 1 < argc) min_slime_chunks = (int)parse_ll_arg(argv[++i], "min-slime-chunks");
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_ll_arg(argv[++i], "limit");
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path) { usage(argv[0]); return 2; }
    if (radius_blocks <= 0 || window_chunks <= 0 || min_slime_chunks < 0) {
        fprintf(stderr, "radius/window must be positive and min slime chunks must be non-negative.\n");
        return 2;
    }

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
        "slime_window_chunks,slime_search_radius_blocks,min_slime_chunks\n",
        line);

    int64_t processed = 0;
    int64_t hits = 0;
    while (fgets(line, sizeof(line), in)) {
        if (limit > 0 && processed >= limit) break;
        trim_line_end(line);
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        int64_t seed = parse_seed_from_csv_line(line);
        int best_x = 0;
        int best_z = 0;
        int best = find_best_slime_window(seed, radius_blocks, window_chunks, &best_x, &best_z);
        if (best >= min_slime_chunks) {
            int min_x = best_x * 16;
            int min_z = best_z * 16;
            int max_x = (best_x + window_chunks) * 16 - 1;
            int max_z = (best_z + window_chunks) * 16 - 1;
            fprintf(out, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                line, best, best_x, best_z, min_x, min_z, max_x, max_z,
                window_chunks, radius_blocks, min_slime_chunks);
            hits++;
        }
        processed++;
        fprintf(stderr, "processed=%" PRId64 " hits=%" PRId64 " seed=%" PRId64 " best_slime=%d\n", processed, hits, seed, best);
        fflush(stderr);
    }

    fclose(in);
    fclose(out);
    fprintf(stderr, "done: processed=%" PRId64 " hits=%" PRId64 " output=%s\n", processed, hits, out_path);
    return 0;
}
