// Filter candidate seeds for a Nether Fortress with adjacent crossroads-like pieces.
// Uses cubiomes. This is a practical approximation for Java 26.2 until exact
// version constants/checkers are available.

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "finders.h"
#include "generator.h"

#define MAX_PIECES 1024

typedef struct { int x, z; } IPos;

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

static const char *piece_type_name(int t) {
    switch (t) {
        case FORTRESS_START: return "FORTRESS_START";
        case BRIDGE_STRAIGHT: return "BRIDGE_STRAIGHT";
        case BRIDGE_CROSSING: return "BRIDGE_CROSSING";
        case BRIDGE_FORTIFIED_CROSSING: return "BRIDGE_FORTIFIED_CROSSING";
        case BRIDGE_STAIRS: return "BRIDGE_STAIRS";
        case BRIDGE_SPAWNER: return "BRIDGE_SPAWNER";
        case BRIDGE_CORRIDOR_ENTRANCE: return "BRIDGE_CORRIDOR_ENTRANCE";
        case CORRIDOR_STRAIGHT: return "CORRIDOR_STRAIGHT";
        case CORRIDOR_CROSSING: return "CORRIDOR_CROSSING";
        case CORRIDOR_TURN_RIGHT: return "CORRIDOR_TURN_RIGHT";
        case CORRIDOR_TURN_LEFT: return "CORRIDOR_TURN_LEFT";
        case CORRIDOR_STAIRS: return "CORRIDOR_STAIRS";
        case CORRIDOR_T_CROSSING: return "CORRIDOR_T_CROSSING";
        case CORRIDOR_NETHER_WART: return "CORRIDOR_NETHER_WART";
        case FORTRESS_END: return "FORTRESS_END";
        default: return "UNKNOWN";
    }
}

static bool is_crossroad_type(int t) {
    return t == BRIDGE_CROSSING || t == BRIDGE_FORTIFIED_CROSSING ||
           t == CORRIDOR_CROSSING || t == CORRIDOR_T_CROSSING;
}

static int interval_gap(int a0, int a1, int b0, int b1) {
    if (a1 < b0) return b0 - a1 - 1;
    if (b1 < a0) return a0 - b1 - 1;
    return 0;
}

static bool pieces_adjacent(const Piece *a, const Piece *b, int max_gap, int *gap_out) {
    int gx = interval_gap(a->bb0.x, a->bb1.x, b->bb0.x, b->bb1.x);
    int gy = interval_gap(a->bb0.y, a->bb1.y, b->bb0.y, b->bb1.y);
    int gz = interval_gap(a->bb0.z, a->bb1.z, b->bb0.z, b->bb1.z);
    int gap = gx;
    if (gy > gap) gap = gy;
    if (gz > gap) gap = gz;
    if (gap_out) *gap_out = gap;
    return gap <= max_gap;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --in candidates.csv --out fortress.csv [options]\n"
        "Options:\n"
        "  --radius-blocks N       search square +/-N in Nether blocks (default 10000)\n"
        "  --circle                require fortress start inside Euclidean radius N\n"
        "  --max-gap-blocks N      max AABB gap between crossroad pieces (default 1)\n"
        "  --mc newest|1_21|1_20   cubiomes version approximation (default newest)\n"
        "  --limit N               process at most N candidate rows (default 0 = all)\n",
        argv0);
}

static bool find_adjacent_crossroads(uint64_t seed, int mc, int radius, int max_gap, bool circle, FILE *out, int64_t signed_seed, const char *mc_name) {
    StructureConfig sc;
    if (!getStructureConfig(Fortress, mc, &sc)) { fprintf(stderr, "Fortress unsupported for mc=%d\n", mc); exit(2); }
    int chunk_radius = (radius + 15) / 16;
    int reg_min = (int)floor((double)(-chunk_radius - sc.regionSize) / sc.regionSize);
    int reg_max = (int)ceil((double)(chunk_radius + sc.regionSize) / sc.regionSize);

    Generator g;
    setupGenerator(&g, mc, 0);
    applySeed(&g, DIM_NETHER, seed);

    Piece pieces[MAX_PIECES];
    int64_t r2 = (int64_t) radius * radius;
    for (int rz = reg_min; rz <= reg_max; rz++) {
        for (int rx = reg_min; rx <= reg_max; rx++) {
            Pos p;
            if (!getStructurePos(Fortress, mc, seed, rx, rz, &p)) continue;
            if (p.x < -radius || p.x > radius || p.z < -radius || p.z > radius) continue;
            if (circle && (int64_t)p.x * p.x + (int64_t)p.z * p.z > r2) continue;
            // Fortress biome viability is cheap enough and avoids impossible starts.
            if (!isViableStructurePos(Fortress, &g, p.x, p.z, 0)) continue;
            int chunkX = p.x >> 4;
            int chunkZ = p.z >> 4;
            int n = getFortressPieces(pieces, MAX_PIECES, mc, seed, chunkX, chunkZ);
            if (n <= 1) continue;
            for (int i = 0; i < n; i++) {
                if (!is_crossroad_type(pieces[i].type)) continue;
                for (int j = i + 1; j < n; j++) {
                    if (!is_crossroad_type(pieces[j].type)) continue;
                    int gap = 0;
                    if (!pieces_adjacent(&pieces[i], &pieces[j], max_gap, &gap)) continue;
                    fprintf(out,
                        "%" PRId64 ",%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d\n",
                        signed_seed, p.x, p.z, chunkX, chunkZ,
                        piece_type_name(pieces[i].type), piece_type_name(pieces[j].type),
                        pieces[i].bb0.x, pieces[i].bb0.y, pieces[i].bb0.z, pieces[i].bb1.x, pieces[i].bb1.y, pieces[i].bb1.z,
                        pieces[j].bb0.x, pieces[j].bb0.y, pieces[j].bb0.z, pieces[j].bb1.x, pieces[j].bb1.y, pieces[j].bb1.z,
                        gap, radius, max_gap, mc_name, rx, rz);
                    return true;
                }
            }
        }
    }
    return false;
}

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL, *mc_name = "newest";
    int radius = 10000;
    int max_gap = 1;
    bool circle = false;
    long limit = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--radius-blocks") && i + 1 < argc) radius = (int)parse_long_arg(argv[++i], "radius-blocks");
        else if (!strcmp(argv[i], "--max-gap-blocks") && i + 1 < argc) max_gap = (int)parse_long_arg(argv[++i], "max-gap-blocks");
        else if (!strcmp(argv[i], "--mc") && i + 1 < argc) mc_name = argv[++i];
        else if (!strcmp(argv[i], "--circle")) circle = true;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = parse_long_arg(argv[++i], "limit");
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path) { usage(argv[0]); return 2; }
    int mc = parse_mc(mc_name);
    FILE *in = fopen(in_path, "r");
    if (!in) { perror(in_path); return 1; }
    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); fclose(in); return 1; }
    char line[4096];
    if (!fgets(line, sizeof(line), in)) { fprintf(stderr, "empty input\n"); return 1; }
    fprintf(out, "seed,fortress_x,fortress_z,fortress_chunk_x,fortress_chunk_z,piece1_type,piece2_type,piece1_bb0x,piece1_bb0y,piece1_bb0z,piece1_bb1x,piece1_bb1y,piece1_bb1z,piece2_bb0x,piece2_bb0y,piece2_bb0z,piece2_bb1x,piece2_bb1y,piece2_bb1z,gap_blocks,radius_blocks,max_gap_blocks,mc_approx,fortress_region_x,fortress_region_z\n");
    long processed = 0, hits = 0;
    while (fgets(line, sizeof(line), in)) {
        if (limit > 0 && processed >= limit) break;
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        int64_t s = parse_seed_from_csv_line(line);
        if (find_adjacent_crossroads((uint64_t)s, mc, radius, max_gap, circle, out, s, mc_name)) hits++;
        processed++;
        if (processed % 100 == 0) { fprintf(stderr, "processed=%ld hits=%ld\n", processed, hits); fflush(stderr); }
    }
    fclose(in); fclose(out);
    fprintf(stderr, "done: processed=%ld hits=%ld output=%s\n", processed, hits, out_path);
    return 0;
}
