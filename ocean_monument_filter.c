// Search/filter candidate seeds for a quad Ocean Monument cluster.
//
// The target cluster is four viable Ocean Monuments whose full configured
// footprints can fit inside one chunk-aligned 17x17 chunk square. This uses
// cubiomes for structure placement and biome viability.
//
// Build example:
//   gcc -O3 -march=native -std=c11 -Wall -Wextra -fwrapv -I../../tools/cubiomes
//       -o ocean_monument_filter ocean_monument_filter.c ../../tools/cubiomes/libcubiomes.a -lm

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "finders.h"
#include "generator.h"

typedef struct {
    int rx;
    int rz;
    int pos_x;
    int pos_z;
    int center_x;
    int center_z;
    int min_x;
    int min_z;
    int max_x;
    int max_z;
} MonumentBox;

typedef struct {
    MonumentBox m[4];
    int window_chunk_x;
    int window_chunk_z;
    int window_min_x;
    int window_min_z;
    int window_max_x;
    int window_max_z;
} QuadMonuments;

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

static int floor_div_i(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r && ((r > 0) != (b > 0))) q--;
    return q;
}

static int ceil_div_i(int a, int b) {
    return -floor_div_i(-a, b);
}

static int max_i(int a, int b) {
    return a > b ? a : b;
}

static int min_i(int a, int b) {
    return a < b ? a : b;
}

static int parse_mc(const char *s) {
    if (!s || !*s || !strcmp(s, "newest") || !strcmp(s, "26.2")) return MC_NEWEST;
    if (!strcmp(s, "1_21") || !strcmp(s, "1.21")) return MC_1_21;
    if (!strcmp(s, "1_20") || !strcmp(s, "1.20")) return MC_1_20;
    fprintf(stderr, "Unsupported --mc %s; use 26.2, newest, 1_21, or 1_20.\n", s);
    exit(2);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --start SEED --count N --out quad_monuments.csv [options]\n"
        "  %s --in candidates.csv --out quad_monuments.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N          origin-centered square +/-N blocks (default 60000)\n"
        "  --window-inside-radius     require the full 17x17 window inside +/-radius\n"
        "  --cluster-window-chunks N  chunk-aligned cluster square size (default 17)\n"
        "  --required-count N         currently only 4 is supported (default 4)\n"
        "  --footprint-blocks N       monument footprint width/depth to contain (default 58)\n"
        "  --center-offset-blocks N   center offset from cubiomes structure pos (default 8)\n"
        "  --mc 26.2|newest|1_21|1_20  cubiomes version approximation (default 26.2)\n"
        "  --limit N                  process at most N input rows/seeds, 0 = all\n"
        "  --max-candidates N         stop after N hits, 0 = no cap\n",
        argv0, argv0);
}

static MonumentBox make_box(Pos p, int rx, int rz, int footprint_blocks, int center_offset_blocks) {
    int half_before = footprint_blocks / 2;
    int half_after = footprint_blocks - half_before - 1;
    int cx = p.x + center_offset_blocks;
    int cz = p.z + center_offset_blocks;
    MonumentBox box;
    box.rx = rx;
    box.rz = rz;
    box.pos_x = p.x;
    box.pos_z = p.z;
    box.center_x = cx;
    box.center_z = cz;
    box.min_x = cx - half_before;
    box.min_z = cz - half_before;
    box.max_x = cx + half_after;
    box.max_z = cz + half_after;
    return box;
}

static bool containing_window_for_quad(
    MonumentBox *m,
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

static bool all_viable(Generator *g, uint64_t seed, MonumentBox *m) {
    applySeed(g, DIM_OVERWORLD, seed);
    for (int i = 0; i < 4; i++) {
        if (!isViableStructurePos(Monument, g, m[i].pos_x, m[i].pos_z, 0)) return false;
    }
    return true;
}

static bool find_quad_monument(
    uint64_t seed,
    int mc,
    int radius,
    int window_chunks,
    int footprint_blocks,
    int center_offset_blocks,
    bool window_inside_radius,
    QuadMonuments *out
) {
    StructureConfig sc;
    if (!getStructureConfig(Monument, mc, &sc)) {
        fprintf(stderr, "Monument unsupported for mc=%d\n", mc);
        exit(2);
    }

    int region_blocks = sc.regionSize * 16;
    int margin = window_chunks * 16 + footprint_blocks + region_blocks;
    int reg_min = floor_div_i(-radius - margin, region_blocks) - 1;
    int reg_max = floor_div_i(radius + margin, region_blocks) + 1;

    Generator g;
    setupGenerator(&g, mc, 0);

    for (int rz = reg_min; rz < reg_max; rz++) {
        for (int rx = reg_min; rx < reg_max; rx++) {
            Pos p00, p10, p01, p11;
            if (!getStructurePos(Monument, mc, seed, rx, rz, &p00)) continue;
            if (!getStructurePos(Monument, mc, seed, rx + 1, rz, &p10)) continue;
            if (!getStructurePos(Monument, mc, seed, rx, rz + 1, &p01)) continue;
            if (!getStructurePos(Monument, mc, seed, rx + 1, rz + 1, &p11)) continue;

            MonumentBox m[4];
            m[0] = make_box(p00, rx, rz, footprint_blocks, center_offset_blocks);
            m[1] = make_box(p10, rx + 1, rz, footprint_blocks, center_offset_blocks);
            m[2] = make_box(p01, rx, rz + 1, footprint_blocks, center_offset_blocks);
            m[3] = make_box(p11, rx + 1, rz + 1, footprint_blocks, center_offset_blocks);

            int wcx = 0, wcz = 0;
            if (!containing_window_for_quad(m, window_chunks, radius, window_inside_radius, &wcx, &wcz)) continue;
            if (!all_viable(&g, seed, m)) continue;

            for (int i = 0; i < 4; i++) out->m[i] = m[i];
            out->window_chunk_x = wcx;
            out->window_chunk_z = wcz;
            out->window_min_x = wcx * 16;
            out->window_min_z = wcz * 16;
            out->window_max_x = (wcx + window_chunks) * 16 - 1;
            out->window_max_z = (wcz + window_chunks) * 16 - 1;
            return true;
        }
    }
    return false;
}

static void trim_line_end(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

static void write_header(FILE *out, const char *input_header) {
    if (input_header && *input_header) {
        fprintf(out, "%s,", input_header);
    } else {
        fprintf(out, "seed,");
    }
    fprintf(out,
        "monument1_center_x,monument1_center_z,monument2_center_x,monument2_center_z,"
        "monument3_center_x,monument3_center_z,monument4_center_x,monument4_center_z,"
        "quad_window_chunk_x,quad_window_chunk_z,quad_window_min_block_x,quad_window_min_block_z,"
        "quad_window_max_block_x,quad_window_max_block_z,cluster_window_chunks,search_radius_blocks,"
        "mc_approx,monument_footprint_blocks,monument_center_offset_blocks\n");
}

static void write_hit(
    FILE *out,
    const char *input_row,
    int64_t seed,
    const QuadMonuments *q,
    int window_chunks,
    int radius,
    const char *mc_name,
    int footprint_blocks,
    int center_offset_blocks
) {
    if (input_row && *input_row) {
        fprintf(out, "%s,", input_row);
    } else {
        fprintf(out, "%" PRId64 ",", seed);
    }
    fprintf(out,
        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d\n",
        q->m[0].center_x, q->m[0].center_z,
        q->m[1].center_x, q->m[1].center_z,
        q->m[2].center_x, q->m[2].center_z,
        q->m[3].center_x, q->m[3].center_z,
        q->window_chunk_x, q->window_chunk_z,
        q->window_min_x, q->window_min_z,
        q->window_max_x, q->window_max_z,
        window_chunks, radius, mc_name, footprint_blocks, center_offset_blocks);
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    const char *mc_name = "26.2";
    int64_t start = 0;
    int64_t count = -1;
    int radius = 60000;
    int window_chunks = 17;
    int required_count = 4;
    int footprint_blocks = 58;
    int center_offset_blocks = 8;
    bool window_inside_radius = false;
    int64_t limit = 0;
    int64_t max_candidates = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_ll_arg(argv[++i], "start");
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = parse_ll_arg(argv[++i], "count");
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_ll_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--cluster-window-chunks") && i + 1 < argc) window_chunks = (int)parse_ll_arg(argv[++i], "cluster-window-chunks");
        else if (!strcmp(argv[i], "--required-count") && i + 1 < argc) required_count = (int)parse_ll_arg(argv[++i], "required-count");
        else if (!strcmp(argv[i], "--footprint-blocks") && i + 1 < argc) footprint_blocks = (int)parse_ll_arg(argv[++i], "footprint-blocks");
        else if (!strcmp(argv[i], "--center-offset-blocks") && i + 1 < argc) center_offset_blocks = (int)parse_ll_arg(argv[++i], "center-offset-blocks");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) mc_name = argv[++i];
        else if (!strcmp(argv[i], "--window-inside-radius")) window_inside_radius = true;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_ll_arg(argv[++i], "limit");
        else if (!strcmp(argv[i], "--max-candidates") && i + 1 < argc) max_candidates = parse_ll_arg(argv[++i], "max-candidates");
        else { usage(argv[0]); return 2; }
    }

    if (!out_path || ((in_path == NULL) == (count < 0))) {
        usage(argv[0]);
        return 2;
    }
    if (required_count != 4) {
        fprintf(stderr, "Only --required-count 4 is currently supported.\n");
        return 2;
    }
    if (radius <= 0 || window_chunks <= 0 || footprint_blocks <= 0) {
        fprintf(stderr, "radius, window chunks, and footprint must be positive.\n");
        return 2;
    }

    int mc = parse_mc(mc_name);
    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); return 1; }

    int64_t processed = 0;
    int64_t hits = 0;
    if (in_path) {
        FILE *in = fopen(in_path, "r");
        if (!in) { perror(in_path); fclose(out); return 1; }
        char line[16384];
        if (!fgets(line, sizeof(line), in)) {
            fprintf(stderr, "empty input\n");
            fclose(in);
            fclose(out);
            return 1;
        }
        trim_line_end(line);
        write_header(out, line);
        while (fgets(line, sizeof(line), in)) {
            if (limit > 0 && processed >= limit) break;
            trim_line_end(line);
            if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
            int64_t s = parse_seed_from_csv_line(line);
            QuadMonuments q;
            if (find_quad_monument((uint64_t)s, mc, radius, window_chunks, footprint_blocks, center_offset_blocks, window_inside_radius, &q)) {
                write_hit(out, line, s, &q, window_chunks, radius, mc_name, footprint_blocks, center_offset_blocks);
                hits++;
                if (max_candidates > 0 && hits >= max_candidates) break;
            }
            processed++;
            if (processed % 1000 == 0) {
                fprintf(stderr, "processed=%" PRId64 " hits=%" PRId64 "\n", processed, hits);
                fflush(stderr);
            }
        }
        fclose(in);
    } else {
        write_header(out, NULL);
        for (int64_t idx = 0; idx < count; idx++) {
            if (limit > 0 && processed >= limit) break;
            int64_t s = start + idx;
            QuadMonuments q;
            if (find_quad_monument((uint64_t)s, mc, radius, window_chunks, footprint_blocks, center_offset_blocks, window_inside_radius, &q)) {
                write_hit(out, NULL, s, &q, window_chunks, radius, mc_name, footprint_blocks, center_offset_blocks);
                hits++;
                if (max_candidates > 0 && hits >= max_candidates) break;
            }
            processed++;
            if (processed % 1000 == 0) {
                fprintf(stderr, "progress: start=%" PRId64 " scanned=%" PRId64 "/%" PRId64 " candidates=%" PRId64 "\n", start, processed, count, hits);
                fflush(stderr);
            }
        }
    }

    fclose(out);
    fprintf(stderr, "done: processed=%" PRId64 " hits=%" PRId64 " output=%s\n", processed, hits, out_path);
    return 0;
}
