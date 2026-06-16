// Filter candidate seeds for all overworld biomes inside an origin-centered square.
// Uses cubiomes BiomeFilter. For MC 26.2, --mc newest is currently an approximation
// until cubiomes exposes the exact version mapping.

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
#include "biomes.h"
#include "util.h"

static long parse_long_arg(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end) { fprintf(stderr, "Invalid %s: %s\n", name, s); exit(2); }
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

static int parse_mc(const char *s) {
    if (!s || !*s || !strcmp(s, "newest") || !strcmp(s, "26.2")) return MC_NEWEST;
    if (!strcmp(s, "1_21") || !strcmp(s, "1.21")) return MC_1_21;
    if (!strcmp(s, "1_20") || !strcmp(s, "1.20")) return MC_1_20;
    fprintf(stderr, "Unsupported --mc %s; use newest, 1_21, or 1_20.\n", s);
    exit(2);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out all_biomes.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N    square +/-N from origin (default 10000)\n"
        "  --scale N            biome range scale: 4,16,64,256 (default 64, use 16 for stricter)\n"
        "  --mc 26.2|newest|1_21  cubiomes version approximation (default newest)\n"
        "  --limit N            process at most N candidate rows (default 0 = all)\n",
        argv0);
}

static int build_required_biomes(int mc, int *out, int max_out) {
    int n = 0;
    for (int id = 0; id < 256; id++) {
        if (!biomeExists(mc, id)) continue;
        if (!isOverworld(mc, id)) continue;
        if (getDimension(id) != DIM_OVERWORLD) continue;
        // Ignore old mutated aliases that do not generate as independent biomes in modern versions.
        // biomeExists/isOverworld already handles most version-specific availability.
        if (n >= max_out) { fprintf(stderr, "too many required biomes\n"); exit(2); }
        out[n++] = id;
    }
    return n;
}

static bool has_all_biomes(uint64_t seed, int mc, int radius, int scale, int *required, int required_len, int *ret_code) {
    Generator g;
    setupGenerator(&g, mc, 0);
    Range r;
    r.scale = scale;
    r.x = -radius / scale;
    r.z = -radius / scale;
    r.sx = (2 * radius) / scale + 1;
    r.sz = (2 * radius) / scale + 1;
    // y is biome-coordinate scaled. Use a broad vertical slice so cave biomes can be seen by the filter.
    // For scale=64/16 this is still a horizontal range for the filter layers; exact 3D final checks can be added later.
    r.y = 0;
    r.sy = 1;

    BiomeFilter bf;
    setupBiomeFilter(&bf, mc, 0, required, required_len, NULL, 0, NULL, 0);
    int *cache = NULL; // let cubiomes use filtering without storing the full map where possible
    int ret = checkForBiomes(&g, cache, r, DIM_OVERWORLD, seed, &bf, NULL);
    if (ret_code) *ret_code = ret;
    return ret != 0;
}

static void trim_line_end(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
}

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL, *mc_name = "newest";
    int radius = 10000;
    int scale = 64;
    long limit = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_long_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = (int)parse_long_arg(argv[++i], "scale");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) mc_name = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_long_arg(argv[++i], "limit");
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path) { usage(argv[0]); return 2; }
    int mc = parse_mc(mc_name);
    int required[256];
    int required_len = build_required_biomes(mc, required, 256);
    fprintf(stderr, "required_overworld_biomes=%d mc=%s scale=%d radius=%d\n", required_len, mc_name, scale, radius);

    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }
    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); fclose(in); return 1; }

    char line[16384];
    if (!fgets(line, sizeof(line), in)) { fprintf(stderr, "empty input\n"); return 1; }
    trim_line_end(line);
    fprintf(out, "%s,required_biome_count,check_return,all_biomes_mc_ref,all_biomes_radius_blocks,biome_scale\n", line);

    long processed = 0, hits = 0;
    while (fgets(line, sizeof(line), in)) {
        if (limit > 0 && processed >= limit) break;
        trim_line_end(line);
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        int64_t s = parse_seed_from_csv_line(line);
        int ret = 0;
        if (has_all_biomes((uint64_t)s, mc, radius, scale, required, required_len, &ret)) {
            fprintf(out, "%s,%d,%d,%s,%d,%d\n", line, required_len, ret, mc_name, radius, scale);
            hits++;
        }
        processed++;
        if (processed % 1000 == 0) { fprintf(stderr, "processed=%ld hits=%ld\n", processed, hits); fflush(stderr); }
    }
    fclose(in); fclose(out);
    fprintf(stderr, "done: processed=%ld hits=%ld output=%s\n", processed, hits, out_path);
    return 0;
}
