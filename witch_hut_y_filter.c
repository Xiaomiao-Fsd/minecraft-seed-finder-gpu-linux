// Find the lowest-Y Witch Hut (Swamp Hut) candidate for each input Minecraft Java seed.
//
// Reads a CSV whose first column is a seed, scans viable Swamp Hut structure
// positions inside an origin-centered radius, keeps the lowest approximate
// surface Y per seed, then writes results sorted by Y ascending.
//
// Y caveat: cubiomes gives structure X/Z exactly, but Overworld terrain height
// here uses cubiomes mapApproxHeight() at the hut chunk center. Treat Y as a
// fast ranking/locator value and do final in-game/Chunkbase validation if exact
// block-level Y is required.

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "finders.h"
#include "generator.h"

typedef struct {
    long long seed;
    int found;
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
    long checked;
    long viable;
} Result;

typedef struct {
    Result *v;
    size_t n;
    size_t cap;
} Results;

static long long parse_ll(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static double parse_double(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !end || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static int parse_seed_line(const char *line, long long *seed) {
    const char *p = line;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '-' && !isdigit((unsigned char)*p)) return 0;
    errno = 0;
    char *end = NULL;
    long long v = strtoll(p, &end, 10);
    if (errno || end == p) return 0;
    if (*end && *end != ',' && *end != '\n' && *end != '\r' && !isspace((unsigned char)*end)) return 0;
    *seed = v;
    return 1;
}

static int floor_div_i(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r && ((r > 0) != (b > 0))) q--;
    return q;
}

static int parse_mc(const char *s) {
    if (!s || !*s || !strcmp(s, "26.2") || !strcmp(s, "newest")) return MC_NEWEST;
    if (!strcmp(s, "1_21") || !strcmp(s, "1.21")) return MC_1_21;
    if (!strcmp(s, "1_20") || !strcmp(s, "1.20")) return MC_1_20;
    if (!strcmp(s, "1_19") || !strcmp(s, "1.19")) return MC_1_19;
    if (!strcmp(s, "1_18") || !strcmp(s, "1.18")) return MC_1_18;
    fprintf(stderr, "Unsupported --mc %s; use 26.2,newest,1_21,1_20,1_19,1_18\n", s);
    exit(2);
}

static void push_result(Results *rs, Result r) {
    if (rs->n == rs->cap) {
        size_t nc = rs->cap ? rs->cap * 2 : 1024;
        Result *nv = (Result *)realloc(rs->v, nc * sizeof(*nv));
        if (!nv) { perror("realloc"); exit(1); }
        rs->v = nv;
        rs->cap = nc;
    }
    rs->v[rs->n++] = r;
}

static int cmp_result(const void *a, const void *b) {
    const Result *ra = (const Result *)a;
    const Result *rb = (const Result *)b;
    if (ra->found != rb->found) return rb->found - ra->found;
    if (!ra->found) return (ra->seed > rb->seed) - (ra->seed < rb->seed);
    if (ra->y < rb->y) return -1;
    if (ra->y > rb->y) return 1;
    if (ra->seed != rb->seed) return (ra->seed > rb->seed) - (ra->seed < rb->seed);
    if (ra->x != rb->x) return ra->x - rb->x;
    return ra->z - rb->z;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out witch_huts_lowest.csv [options]\n"
        "\n"
        "Options:\n"
        "  --radius-blocks N   search square +/-N blocks from origin (default 60000)\n"
        "  --circle            use Euclidean radius instead of square bounds\n"
        "  --mc VERSION        26.2|newest|1_21|1_20|1_19|1_18 (default 26.2)\n"
        "  --no-biome-check    include structure attempts without Swamp biome viability\n"
        "  --negative-y-only   only keep huts with hut_y_approx < 0\n"
        "  --max-y N           only keep huts with hut_y_approx <= N\n"
        "  --include-missing   output seeds with no hut matching filters, sorted after hits\n"
        "  --limit N           process at most N input seed rows, 0 = all (default 0)\n"
        "  --top N             write at most N found rows after sorting, 0 = all (default 0)\n"
        "  --progress N        print progress every N seeds, 0 = silent (default 100)\n"
        "\n"
        "Output is one lowest-Y hut per seed, sorted by hut_y_approx ascending.\n"
        "Y is cubiomes mapApproxHeight() at the hut chunk center, not full block-level terrain.\n",
        argv0);
}

static Result scan_seed(
    long long signed_seed,
    int mc,
    int radius,
    int circle,
    int require_biome,
    int negative_y_only,
    int max_y_enabled,
    float max_y
) {
    Result best;
    memset(&best, 0, sizeof(best));
    best.seed = signed_seed;

    StructureConfig sc;
    if (!getStructureConfig(Swamp_Hut, mc, &sc)) {
        fprintf(stderr, "Swamp_Hut unsupported for mc=%d\n", mc);
        exit(2);
    }

    uint64_t seed = (uint64_t)signed_seed;
    int region_blocks = sc.regionSize * 16;
    int reg_min = floor_div_i(-radius, region_blocks) - 1;
    int reg_max = floor_div_i(radius, region_blocks) + 1;
    int64_t r2 = (int64_t)radius * (int64_t)radius;

    Generator g;
    setupGenerator(&g, mc, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, seed);

    for (int rz = reg_min; rz <= reg_max; rz++) {
        for (int rx = reg_min; rx <= reg_max; rx++) {
            Pos p;
            if (!getStructurePos(Swamp_Hut, mc, seed, rx, rz, &p)) continue;
            if (p.x < -radius || p.x > radius || p.z < -radius || p.z > radius) continue;
            if (circle && (int64_t)p.x * p.x + (int64_t)p.z * p.z > r2) continue;
            best.checked++;

            if (require_biome && !isViableStructurePos(Swamp_Hut, &g, p.x, p.z, 0)) continue;
            best.viable++;

            int chunk_x = p.x >> 4;
            int chunk_z = p.z >> 4;
            int sample_x = chunk_x * 16 + 8;
            int sample_z = chunk_z * 16 + 8;
            float y = 0.0f;
            int biome_id = -1;
            if (mapApproxHeight(&y, &biome_id, &g, &sn, sample_x >> 2, sample_z >> 2, 1, 1) != 0) continue;
            if (negative_y_only && !(y < 0.0f)) continue;
            if (max_y_enabled && y > max_y) continue;

            if (!best.found || y < best.y || (y == best.y && (p.x < best.x || (p.x == best.x && p.z < best.z)))) {
                best.found = 1;
                best.x = p.x;
                best.z = p.z;
                best.chunk_x = chunk_x;
                best.chunk_z = chunk_z;
                best.region_x = rx;
                best.region_z = rz;
                best.sample_x = sample_x;
                best.sample_z = sample_z;
                best.y = y;
                best.y_block = (int)floorf(y + 0.5f);
                best.biome_id = biome_id;
            }
        }
    }
    return best;
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    const char *mc_name = "26.2";
    int radius = 60000;
    int circle = 0;
    int require_biome = 1;
    int negative_y_only = 0;
    int max_y_enabled = 0;
    float max_y = 0.0f;
    int include_missing = 0;
    long limit = 0;
    long top = 0;
    long progress = 100;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_ll(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) mc_name = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = (long)parse_ll(argv[++i], "limit");
        else if (!strcmp(argv[i], "--top") && i + 1 < argc) top = (long)parse_ll(argv[++i], "top");
        else if (!strcmp(argv[i], "--progress") && i + 1 < argc) progress = (long)parse_ll(argv[++i], "progress");
        else if (!strcmp(argv[i], "--max-y") && i + 1 < argc) { max_y = (float)parse_double(argv[++i], "max-y"); max_y_enabled = 1; }
        else if (!strcmp(argv[i], "--circle")) circle = 1;
        else if (!strcmp(argv[i], "--no-biome-check")) require_biome = 0;
        else if (!strcmp(argv[i], "--negative-y-only")) negative_y_only = 1;
        else if (!strcmp(argv[i], "--include-missing")) include_missing = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path || radius <= 0 || limit < 0 || top < 0 || progress < 0) {
        usage(argv[0]);
        return 2;
    }

    int mc = parse_mc(mc_name);
    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }

    Results results = {0};
    char line[16384];
    long processed = 0, hits = 0, skipped = 0;
    while (fgets(line, sizeof(line), in)) {
        long long seed = 0;
        if (!parse_seed_line(line, &seed)) { skipped++; continue; }
        if (limit > 0 && processed >= limit) break;
        Result r = scan_seed(seed, mc, radius, circle, require_biome, negative_y_only, max_y_enabled, max_y);
        if (r.found) { hits++; push_result(&results, r); }
        else if (include_missing) { push_result(&results, r); }
        processed++;
        if (progress > 0 && processed % progress == 0) {
            fprintf(stderr, "processed=%ld hits=%ld skipped=%ld\n", processed, hits, skipped);
            fflush(stderr);
        }
    }
    fclose(in);

    qsort(results.v, results.n, sizeof(results.v[0]), cmp_result);

    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); free(results.v); return 1; }
    fprintf(out,
        "seed,hut_x,hut_y_approx,hut_y_approx_block,hut_z,"
        "hut_chunk_x,hut_chunk_z,hut_region_x,hut_region_z,"
        "height_sample_x,height_sample_z,surface_biome_id,"
        "checked_structure_positions,viable_huts_in_radius,"
        "radius_blocks,mc_approx,method,note\n");

    const char *missing_note = negative_y_only
        ? "no negative-Y hut found in radius"
        : (max_y_enabled ? "no hut at or below max-y found in radius" : "no hut found in radius");

    long written = 0;
    for (size_t i = 0; i < results.n; i++) {
        const Result *r = &results.v[i];
        if (top > 0 && r->found && written >= top) break;
        if (!r->found) {
            fprintf(out, "%lld,,,,,,,,,,,,%ld,%ld,%d,%s,missing,%s\n",
                    r->seed, r->checked, r->viable, radius, mc_name, missing_note);
            continue;
        }
        fprintf(out,
            "%lld,%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%ld,%d,%s,"
            "cubiomes_structure_pos_plus_mapApproxHeight_chunk_center,"
            "approximate Y; validate in-game for exact block-level terrain\n",
            r->seed, r->x, r->y, r->y_block, r->z,
            r->chunk_x, r->chunk_z, r->region_x, r->region_z,
            r->sample_x, r->sample_z, r->biome_id,
            r->checked, r->viable, radius, mc_name);
        written++;
    }
    fclose(out);

    fprintf(stderr, "done: processed=%ld hits=%ld skipped_non_seed_lines=%ld wrote=%ld output=%s\n",
            processed, hits, skipped, written, out_path);
    if (results.n > 0 && results.v[0].found) {
        fprintf(stderr, "best: seed=%lld hut=(%d,%d,%d) approx_y=%.3f\n",
                results.v[0].seed, results.v[0].x, results.v[0].y_block, results.v[0].z, results.v[0].y);
    }
    free(results.v);
    return 0;
}
