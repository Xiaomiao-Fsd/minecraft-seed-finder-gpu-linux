// Minimal CUDA port of the cubiomes 1.18+ overworld biome-noise path.
// Biome parameter matching uses the official Minecraft Java 26.2
// data-generator biome parameter report.
//
// Portions are derived from cubiomes (MIT License, Copyright 2020 Cubitect).
// See THIRD_PARTY_NOTICES.md.

#pragma once

#include <math.h>
#include <stdint.h>

enum {
    BIOME_NONE = -1,
    BIOME_OCEAN = 0,
    BIOME_PLAINS = 1,
    BIOME_DESERT = 2,
    BIOME_MOUNTAINS = 3,
    BIOME_FOREST = 4,
    BIOME_TAIGA = 5,
    BIOME_SWAMP = 6,
    BIOME_RIVER = 7,
    BIOME_FROZEN_OCEAN = 10,
    BIOME_FROZEN_RIVER = 11,
    BIOME_SNOWY_TUNDRA = 12,
    BIOME_MUSHROOM_FIELDS = 14,
    BIOME_BEACH = 16,
    BIOME_JUNGLE = 21,
    BIOME_JUNGLE_EDGE = 23,
    BIOME_DEEP_OCEAN = 24,
    BIOME_STONE_SHORE = 25,
    BIOME_SNOWY_BEACH = 26,
    BIOME_BIRCH_FOREST = 27,
    BIOME_DARK_FOREST = 29,
    BIOME_SNOWY_TAIGA = 30,
    BIOME_GIANT_TREE_TAIGA = 32,
    BIOME_WOODED_MOUNTAINS = 34,
    BIOME_SAVANNA = 35,
    BIOME_SAVANNA_PLATEAU = 36,
    BIOME_BADLANDS = 37,
    BIOME_WOODED_BADLANDS_PLATEAU = 38,
    BIOME_WARM_OCEAN = 44,
    BIOME_LUKEWARM_OCEAN = 45,
    BIOME_COLD_OCEAN = 46,
    BIOME_DEEP_WARM_OCEAN = 47,
    BIOME_DEEP_LUKEWARM_OCEAN = 48,
    BIOME_DEEP_COLD_OCEAN = 49,
    BIOME_DEEP_FROZEN_OCEAN = 50,
    BIOME_SUNFLOWER_PLAINS = 129,
    BIOME_GRAVELLY_MOUNTAINS = 131,
    BIOME_FLOWER_FOREST = 132,
    BIOME_ICE_SPIKES = 140,
    BIOME_TALL_BIRCH_FOREST = 155,
    BIOME_GIANT_SPRUCE_TAIGA = 160,
    BIOME_SHATTERED_SAVANNA = 163,
    BIOME_ERODED_BADLANDS = 165,
    BIOME_BAMBOO_JUNGLE = 168,
    BIOME_DRIPSTONE_CAVES = 174,
    BIOME_LUSH_CAVES = 175,
    BIOME_MEADOW = 177,
    BIOME_GROVE = 178,
    BIOME_SNOWY_SLOPES = 179,
    BIOME_JAGGED_PEAKS = 180,
    BIOME_FROZEN_PEAKS = 181,
    BIOME_STONY_PEAKS = 182,
    BIOME_DEEP_DARK = 183,
    BIOME_MANGROVE_SWAMP = 184,
    BIOME_CHERRY_GROVE = 185,
    BIOME_PALE_GARDEN = 186,
    BIOME_SULFUR_CAVES = 187,
};

enum {
    NP_TEMPERATURE = 0,
    NP_HUMIDITY = 1,
    NP_CONTINENTALNESS = 2,
    NP_EROSION = 3,
    NP_SHIFT = 4,
    NP_DEPTH = NP_SHIFT,
    NP_WEIRDNESS = 5,
    NP_MAX = 6,
};

enum {
    SAMPLE_NO_SHIFT = 0x1,
    SAMPLE_NO_DEPTH = 0x2,
    SAMPLE_NO_BIOME = 0x4,
};

struct XoroshiroCuda {
    uint64_t lo;
    uint64_t hi;
};

struct PerlinNoiseCuda {
    uint8_t d[257];
    uint8_t h2;
    double a;
    double b;
    double c;
    double amplitude;
    double lacunarity;
    double d2;
    double t2;
};

struct OctaveNoiseCuda {
    int octcnt;
    PerlinNoiseCuda *octaves;
};

struct DoublePerlinNoiseCuda {
    double amplitude;
    OctaveNoiseCuda octA;
    OctaveNoiseCuda octB;
};

struct SplineCuda {
    int len;
    int typ;
    float loc[12];
    float der[12];
    SplineCuda *val[12];
};

struct FixSplineCuda {
    int len;
    float val;
};

struct SplineStackCuda {
    SplineCuda stack[42];
    FixSplineCuda fstack[151];
    int len;
    int flen;
};

struct BiomeNoiseCuda {
    DoublePerlinNoiseCuda climate[NP_MAX];
    PerlinNoiseCuda oct[46];
    SplineCuda *sp;
    SplineStackCuda ss;
    int nptype;
};

struct BiomeTreeCuda {
    const uint32_t *steps;
    const int32_t *param;
    const uint64_t *nodes;
    uint32_t order;
    uint32_t len;
};

static __device__ const uint64_t md5_octave_n_cuda[13][2] = {
    {0xb198de63a8012672ULL, 0x7b84cad43ef7b5a8ULL},
    {0x0fd787bfbc403ec3ULL, 0x74a4a31ca21b48b8ULL},
    {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL},
    {0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL},
    {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL},
    {0xf11268128982754fULL, 0x257a1d670430b0aaULL},
    {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL},
    {0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL},
    {0xbd90d5377ba1b762ULL, 0xc07317d419a7548dULL},
    {0x53d39c6752dac858ULL, 0xbcd1c5a80ab65b3eULL},
    {0xb4a24d7a84e7677bULL, 0x023ff9668e89b5c4ULL},
    {0xdffa22b534c5f608ULL, 0xb9b67517d3665ca9ULL},
    {0xd50708086cef4d7cULL, 0x6e1651ecc7f43309ULL},
};

static __device__ const double lacuna_ini_cuda[13] = {
    1.0, 0.5, 0.25, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0,
    1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0, 1.0 / 1024.0,
    1.0 / 2048.0, 1.0 / 4096.0,
};

static __device__ const double persist_ini_cuda[10] = {
    0.0, 1.0, 2.0 / 3.0, 4.0 / 7.0, 8.0 / 15.0, 16.0 / 31.0,
    32.0 / 63.0, 64.0 / 127.0, 128.0 / 255.0, 256.0 / 511.0,
};

#include "minecraft_26_2_biome_params_cuda.h"

__device__ __forceinline__ uint64_t rotl64_cuda(uint64_t x, uint8_t b) {
    return (x << b) | (x >> (64 - b));
}

__device__ __forceinline__ void xset_seed_cuda(XoroshiroCuda *xr, uint64_t value) {
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    xr->lo = l ^ (l >> 31);
    xr->hi = h ^ (h >> 31);
}

__device__ __forceinline__ uint64_t xnext_long_cuda(XoroshiroCuda *xr) {
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64_cuda(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64_cuda(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64_cuda(h, 28);
    return n;
}

__device__ __forceinline__ int xnext_int_cuda(XoroshiroCuda *xr, uint32_t n) {
    uint64_t r = (xnext_long_cuda(xr) & 0xffffffffULL) * n;
    if ((uint32_t)r < n) {
        while ((uint32_t)r < (~n + 1U) % n) {
            r = (xnext_long_cuda(xr) & 0xffffffffULL) * n;
        }
    }
    return (int)(r >> 32);
}

__device__ __forceinline__ double xnext_double_cuda(XoroshiroCuda *xr) {
    return (xnext_long_cuda(xr) >> (64 - 53)) * 1.1102230246251565E-16;
}

__device__ __forceinline__ double lerp_cuda(double part, double from, double to) {
    return from + part * (to - from);
}

__device__ __forceinline__ double indexed_lerp_cuda(uint8_t idx, double a, double b, double c) {
    switch (idx & 0xf) {
    case 0: return a + b;
    case 1: return -a + b;
    case 2: return a - b;
    case 3: return -a - b;
    case 4: return a + c;
    case 5: return -a + c;
    case 6: return a - c;
    case 7: return -a - c;
    case 8: return b + c;
    case 9: return -b + c;
    case 10: return b - c;
    case 11: return -b - c;
    case 12: return a + b;
    case 13: return -b + c;
    case 14: return -a + b;
    default: return -b - c;
    }
}

__device__ void xperlin_init_cuda(PerlinNoiseCuda *noise, XoroshiroCuda *xr) {
    noise->a = xnext_double_cuda(xr) * 256.0;
    noise->b = xnext_double_cuda(xr) * 256.0;
    noise->c = xnext_double_cuda(xr) * 256.0;
    noise->amplitude = 1.0;
    noise->lacunarity = 1.0;
    for (int i = 0; i < 256; i++) noise->d[i] = (uint8_t)i;
    for (int i = 0; i < 256; i++) {
        int j = xnext_int_cuda(xr, 256U - (uint32_t)i) + i;
        uint8_t n = noise->d[i];
        noise->d[i] = noise->d[j];
        noise->d[j] = n;
    }
    noise->d[256] = noise->d[0];
    double i2 = floor(noise->b);
    double d2 = noise->b - i2;
    noise->h2 = (uint8_t)((int)i2);
    noise->d2 = d2;
    noise->t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
}

__device__ double sample_perlin_cuda(const PerlinNoiseCuda *noise, double d1, double d2, double d3, double yamp, double ymin) {
    uint8_t h1, h2, h3;
    double t1, t2, t3;

    if (d2 == 0.0) {
        d2 = noise->d2;
        h2 = noise->h2;
        t2 = noise->t2;
    } else {
        d2 += noise->b;
        double i2 = floor(d2);
        d2 -= i2;
        h2 = (uint8_t)((int)i2);
        t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
    }

    d1 += noise->a;
    d3 += noise->c;
    double i1 = floor(d1);
    double i3 = floor(d3);
    d1 -= i1;
    d3 -= i3;
    h1 = (uint8_t)((int)i1);
    h3 = (uint8_t)((int)i3);
    t1 = d1 * d1 * d1 * (d1 * (d1 * 6.0 - 15.0) + 10.0);
    t3 = d3 * d3 * d3 * (d3 * (d3 * 6.0 - 15.0) + 10.0);

    if (yamp) {
        double yclamp = ymin < d2 ? ymin : d2;
        d2 -= floor(yclamp / yamp) * yamp;
    }

    const uint8_t *idx = noise->d;
    uint8_t v1a = idx[h1], v1b = idx[(uint8_t)(h1 + 1)];
    v1a += h2; v1b += h2;
    uint8_t v2a = idx[v1a], v2b = idx[(uint8_t)(v1a + 1)];
    uint8_t v3a = idx[v1b], v3b = idx[(uint8_t)(v1b + 1)];
    v2a += h3; v2b += h3; v3a += h3; v3b += h3;
    uint8_t v4a = idx[v2a], v4b = idx[(uint8_t)(v2a + 1)];
    uint8_t v5a = idx[v2b], v5b = idx[(uint8_t)(v2b + 1)];
    uint8_t v6a = idx[v3a], v6b = idx[(uint8_t)(v3a + 1)];
    uint8_t v7a = idx[v3b], v7b = idx[(uint8_t)(v3b + 1)];

    double l1 = indexed_lerp_cuda(v4a, d1, d2, d3);
    double l5 = indexed_lerp_cuda(v4b, d1, d2, d3 - 1);
    double l2 = indexed_lerp_cuda(v6a, d1 - 1, d2, d3);
    double l6 = indexed_lerp_cuda(v6b, d1 - 1, d2, d3 - 1);
    double l3 = indexed_lerp_cuda(v5a, d1, d2 - 1, d3);
    double l7 = indexed_lerp_cuda(v5b, d1, d2 - 1, d3 - 1);
    double l4 = indexed_lerp_cuda(v7a, d1 - 1, d2 - 1, d3);
    double l8 = indexed_lerp_cuda(v7b, d1 - 1, d2 - 1, d3 - 1);

    l1 = lerp_cuda(t1, l1, l2);
    l3 = lerp_cuda(t1, l3, l4);
    l5 = lerp_cuda(t1, l5, l6);
    l7 = lerp_cuda(t1, l7, l8);
    l1 = lerp_cuda(t2, l1, l3);
    l5 = lerp_cuda(t2, l5, l7);
    return lerp_cuda(t3, l1, l5);
}

__device__ int xoctave_init_cuda(OctaveNoiseCuda *noise, XoroshiroCuda *xr, PerlinNoiseCuda *octaves, const double *amplitudes, int omin, int len, int nmax) {
    double lacuna = lacuna_ini_cuda[-omin];
    double persist = persist_ini_cuda[len];
    uint64_t xlo = xnext_long_cuda(xr);
    uint64_t xhi = xnext_long_cuda(xr);
    int n = 0;
    for (int i = 0; i < len && n != nmax; i++, lacuna *= 2.0, persist *= 0.5) {
        if (amplitudes[i] == 0.0) continue;
        XoroshiroCuda pxr;
        int md5_idx = 12 + omin + i;
        pxr.lo = xlo ^ md5_octave_n_cuda[md5_idx][0];
        pxr.hi = xhi ^ md5_octave_n_cuda[md5_idx][1];
        xperlin_init_cuda(&octaves[n], &pxr);
        octaves[n].amplitude = amplitudes[i] * persist;
        octaves[n].lacunarity = lacuna;
        n++;
    }
    noise->octaves = octaves;
    noise->octcnt = n;
    return n;
}

__device__ int xdouble_perlin_init_cuda(DoublePerlinNoiseCuda *noise, XoroshiroCuda *xr, PerlinNoiseCuda *octaves, const double *amplitudes, int omin, int len, int nmax) {
    int n = 0;
    int na = -1, nb = -1;
    if (nmax > 0) {
        na = (nmax + 1) >> 1;
        nb = nmax - na;
    }
    n += xoctave_init_cuda(&noise->octA, xr, octaves + n, amplitudes, omin, len, na);
    n += xoctave_init_cuda(&noise->octB, xr, octaves + n, amplitudes, omin, len, nb);
    while (len > 0 && amplitudes[len - 1] == 0.0) len--;
    int first = 0;
    while (first < len && amplitudes[first] == 0.0) {
        first++;
        len--;
    }
    const double amp_ini[10] = {
        0.0, 5.0 / 6.0, 10.0 / 9.0, 15.0 / 12.0, 20.0 / 15.0,
        25.0 / 18.0, 30.0 / 21.0, 35.0 / 24.0, 40.0 / 27.0, 45.0 / 30.0,
    };
    (void)first;
    noise->amplitude = amp_ini[len];
    return n;
}

__device__ double sample_octave_cuda(const OctaveNoiseCuda *noise, double x, double y, double z) {
    double v = 0.0;
    for (int i = 0; i < noise->octcnt; i++) {
        const PerlinNoiseCuda *p = noise->octaves + i;
        double lf = p->lacunarity;
        double pv = sample_perlin_cuda(p, x * lf, y * lf, z * lf, 0.0, 0.0);
        v += p->amplitude * pv;
    }
    return v;
}

__device__ double sample_double_perlin_cuda(const DoublePerlinNoiseCuda *noise, double x, double y, double z) {
    const double f = 337.0 / 331.0;
    double v = 0.0;
    v += sample_octave_cuda(&noise->octA, x, y, z);
    v += sample_octave_cuda(&noise->octB, x * f, y * f, z * f);
    return v * noise->amplitude;
}

__device__ void add_spline_val_cuda(SplineCuda *rsp, float loc, SplineCuda *val, float der) {
    int i = rsp->len++;
    rsp->loc[i] = loc;
    rsp->val[i] = val;
    rsp->der[i] = der;
}

__device__ SplineCuda *create_fix_spline_cuda(SplineStackCuda *ss, float val) {
    FixSplineCuda *sp = &ss->fstack[ss->flen++];
    sp->len = 1;
    sp->val = val;
    return (SplineCuda *)sp;
}

__device__ float get_offset_value_cuda(float weirdness, float continentalness) {
    float f0 = 1.0F - (1.0F - continentalness) * 0.5F;
    float f1 = 0.5F * (1.0F - continentalness);
    float f2 = (weirdness + 1.17F) * 0.46082947F;
    float off = f2 * f0 - f1;
    if (weirdness < -0.7F) return off > -0.2222F ? off : -0.2222F;
    return off > 0.0F ? off : 0.0F;
}

__device__ SplineCuda *new_spline_cuda(SplineStackCuda *ss, int typ) {
    SplineCuda *sp = &ss->stack[ss->len++];
    sp->len = 0;
    sp->typ = typ;
    return sp;
}

__device__ SplineCuda *create_spline_38219_cuda(SplineStackCuda *ss, float f, int bl) {
    enum { SP_RIDGES = 2 };
    SplineCuda *sp = new_spline_cuda(ss, SP_RIDGES);
    float i = get_offset_value_cuda(-1.0F, f);
    float k = get_offset_value_cuda(1.0F, f);
    float l = 1.0F - (1.0F - f) * 0.5F;
    float u = 0.5F * (1.0F - f);
    l = u / (0.46082947F * l) - 1.17F;
    if (-0.65F < l && l < 1.0F) {
        u = get_offset_value_cuda(-0.65F, f);
        float p = get_offset_value_cuda(-0.75F, f);
        float q = (p - i) * 4.0F;
        float r = get_offset_value_cuda(l, f);
        float s = (k - r) / (1.0F - l);
        add_spline_val_cuda(sp, -1.0F, create_fix_spline_cuda(ss, i), q);
        add_spline_val_cuda(sp, -0.75F, create_fix_spline_cuda(ss, p), 0);
        add_spline_val_cuda(sp, -0.65F, create_fix_spline_cuda(ss, u), 0);
        add_spline_val_cuda(sp, l - 0.01F, create_fix_spline_cuda(ss, r), 0);
        add_spline_val_cuda(sp, l, create_fix_spline_cuda(ss, r), s);
        add_spline_val_cuda(sp, 1.0F, create_fix_spline_cuda(ss, k), s);
    } else {
        u = (k - i) * 0.5F;
        if (bl) {
            add_spline_val_cuda(sp, -1.0F, create_fix_spline_cuda(ss, i > 0.2F ? i : 0.2F), 0);
            add_spline_val_cuda(sp, 0.0F, create_fix_spline_cuda(ss, (float)lerp_cuda(0.5, i, k)), u);
        } else {
            add_spline_val_cuda(sp, -1.0F, create_fix_spline_cuda(ss, i), u);
        }
        add_spline_val_cuda(sp, 1.0F, create_fix_spline_cuda(ss, k), u);
    }
    return sp;
}

__device__ SplineCuda *create_flat_offset_spline_cuda(SplineStackCuda *ss, float f, float g, float h, float i, float j, float k) {
    enum { SP_RIDGES = 2 };
    SplineCuda *sp = new_spline_cuda(ss, SP_RIDGES);
    float l = 0.5F * (g - f); if (l < k) l = k;
    float m = 5.0F * (h - g);
    add_spline_val_cuda(sp, -1.0F, create_fix_spline_cuda(ss, f), l);
    add_spline_val_cuda(sp, -0.4F, create_fix_spline_cuda(ss, g), l < m ? l : m);
    add_spline_val_cuda(sp, 0.0F, create_fix_spline_cuda(ss, h), m);
    add_spline_val_cuda(sp, 0.4F, create_fix_spline_cuda(ss, i), 2.0F * (i - h));
    add_spline_val_cuda(sp, 1.0F, create_fix_spline_cuda(ss, j), 0.7F * (j - i));
    return sp;
}

__device__ SplineCuda *create_land_spline_cuda(SplineStackCuda *ss, float f, float g, float h, float i, float j, float k, int bl) {
    enum { SP_EROSION = 1, SP_RIDGES = 2 };
    SplineCuda *sp1 = create_spline_38219_cuda(ss, (float)lerp_cuda(i, 0.6F, 1.5F), bl);
    SplineCuda *sp2 = create_spline_38219_cuda(ss, (float)lerp_cuda(i, 0.6F, 1.0F), bl);
    SplineCuda *sp3 = create_spline_38219_cuda(ss, i, bl);
    const float ih = 0.5F * i;
    SplineCuda *sp4 = create_flat_offset_spline_cuda(ss, f - 0.15F, ih, ih, ih, i * 0.6F, 0.5F);
    SplineCuda *sp5 = create_flat_offset_spline_cuda(ss, f, j * i, g * i, ih, i * 0.6F, 0.5F);
    SplineCuda *sp6 = create_flat_offset_spline_cuda(ss, f, j, j, g, h, 0.5F);
    SplineCuda *sp7 = create_flat_offset_spline_cuda(ss, f, j, j, g, h, 0.5F);
    SplineCuda *sp8 = new_spline_cuda(ss, SP_RIDGES);
    add_spline_val_cuda(sp8, -1.0F, create_fix_spline_cuda(ss, f), 0.0F);
    add_spline_val_cuda(sp8, -0.4F, sp6, 0.0F);
    add_spline_val_cuda(sp8, 0.0F, create_fix_spline_cuda(ss, h + 0.07F), 0.0F);
    SplineCuda *sp9 = create_flat_offset_spline_cuda(ss, -0.02F, k, k, g, h, 0.0F);
    SplineCuda *sp = new_spline_cuda(ss, SP_EROSION);
    add_spline_val_cuda(sp, -0.85F, sp1, 0.0F);
    add_spline_val_cuda(sp, -0.7F, sp2, 0.0F);
    add_spline_val_cuda(sp, -0.4F, sp3, 0.0F);
    add_spline_val_cuda(sp, -0.35F, sp4, 0.0F);
    add_spline_val_cuda(sp, -0.1F, sp5, 0.0F);
    add_spline_val_cuda(sp, 0.2F, sp6, 0.0F);
    if (bl) {
        add_spline_val_cuda(sp, 0.4F, sp7, 0.0F);
        add_spline_val_cuda(sp, 0.45F, sp8, 0.0F);
        add_spline_val_cuda(sp, 0.55F, sp8, 0.0F);
        add_spline_val_cuda(sp, 0.58F, sp7, 0.0F);
    }
    add_spline_val_cuda(sp, 0.7F, sp9, 0.0F);
    return sp;
}

__device__ float get_spline_cuda(const SplineCuda *sp, const float *vals) {
    if (sp->len == 1) return ((const FixSplineCuda *)sp)->val;
    float f = vals[sp->typ];
    int i = 0;
    for (; i < sp->len; i++) if (sp->loc[i] >= f) break;
    if (i == 0 || i == sp->len) {
        if (i) i--;
        return get_spline_cuda(sp->val[i], vals) + sp->der[i] * (f - sp->loc[i]);
    }
    const SplineCuda *sp1 = sp->val[i - 1];
    const SplineCuda *sp2 = sp->val[i];
    float g = sp->loc[i - 1];
    float h = sp->loc[i];
    float k = (f - g) / (h - g);
    float l = sp->der[i - 1];
    float m = sp->der[i];
    float n = get_spline_cuda(sp1, vals);
    float o = get_spline_cuda(sp2, vals);
    float p = l * (h - g) - (o - n);
    float q = -m * (h - g) + (o - n);
    return (float)lerp_cuda(k, n, o) + k * (1.0F - k) * (float)lerp_cuda(k, p, q);
}

__device__ void init_biome_noise_cuda(BiomeNoiseCuda *bn) {
    SplineStackCuda *ss = &bn->ss;
    ss->len = 0;
    ss->flen = 0;
    SplineCuda *sp = new_spline_cuda(ss, 0);
    SplineCuda *sp1 = create_land_spline_cuda(ss, -0.15F, 0.00F, 0.0F, 0.1F, 0.00F, -0.03F, 0);
    SplineCuda *sp2 = create_land_spline_cuda(ss, -0.10F, 0.03F, 0.1F, 0.1F, 0.01F, -0.03F, 0);
    SplineCuda *sp3 = create_land_spline_cuda(ss, -0.10F, 0.03F, 0.1F, 0.7F, 0.01F, -0.03F, 1);
    SplineCuda *sp4 = create_land_spline_cuda(ss, -0.05F, 0.03F, 0.1F, 1.0F, 0.01F, 0.01F, 1);
    add_spline_val_cuda(sp, -1.10F, create_fix_spline_cuda(ss, 0.044F), 0.0F);
    add_spline_val_cuda(sp, -1.02F, create_fix_spline_cuda(ss, -0.2222F), 0.0F);
    add_spline_val_cuda(sp, -0.51F, create_fix_spline_cuda(ss, -0.2222F), 0.0F);
    add_spline_val_cuda(sp, -0.44F, create_fix_spline_cuda(ss, -0.12F), 0.0F);
    add_spline_val_cuda(sp, -0.18F, create_fix_spline_cuda(ss, -0.12F), 0.0F);
    add_spline_val_cuda(sp, -0.16F, sp1, 0.0F);
    add_spline_val_cuda(sp, -0.15F, sp1, 0.0F);
    add_spline_val_cuda(sp, -0.10F, sp2, 0.0F);
    add_spline_val_cuda(sp, 0.25F, sp3, 0.0F);
    add_spline_val_cuda(sp, 1.00F, sp4, 0.0F);
    bn->sp = sp;
    bn->nptype = -1;
}

__device__ int init_climate_seed_cuda(DoublePerlinNoiseCuda *dpn, PerlinNoiseCuda *oct, uint64_t xlo, uint64_t xhi, int nptype) {
    XoroshiroCuda pxr;
    switch (nptype) {
    case NP_SHIFT: {
        const double amp[] = {1, 1, 1, 0};
        pxr.lo = xlo ^ 0x080518cf6af25384ULL;
        pxr.hi = xhi ^ 0x3f3dfb40a54febd5ULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -3, 4, -1);
    }
    case NP_TEMPERATURE: {
        const double amp[] = {1.5, 0, 1, 0, 0, 0};
        pxr.lo = xlo ^ 0x5c7e6b29735f0d7fULL;
        pxr.hi = xhi ^ 0xf7d86f1bbc734988ULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -10, 6, -1);
    }
    case NP_HUMIDITY: {
        const double amp[] = {1, 1, 0, 0, 0, 0};
        pxr.lo = xlo ^ 0x81bb4d22e8dc168eULL;
        pxr.hi = xhi ^ 0xf1c8b4bea16303cdULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -8, 6, -1);
    }
    case NP_CONTINENTALNESS: {
        const double amp[] = {1, 1, 2, 2, 2, 1, 1, 1, 1};
        pxr.lo = xlo ^ 0x83886c9d0ae3a662ULL;
        pxr.hi = xhi ^ 0xafa638a61b42e8adULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -9, 9, -1);
    }
    case NP_EROSION: {
        const double amp[] = {1, 1, 0, 1, 1};
        pxr.lo = xlo ^ 0xd02491e6058f6fd8ULL;
        pxr.hi = xhi ^ 0x4792512c94c17a80ULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -9, 5, -1);
    }
    default: {
        const double amp[] = {1, 2, 1, 0, 0, 0};
        pxr.lo = xlo ^ 0xefc8ef4d36102b34ULL;
        pxr.hi = xhi ^ 0x1beeeb324a0f24eaULL;
        return xdouble_perlin_init_cuda(dpn, &pxr, oct, amp, -7, 6, -1);
    }
    }
}

__device__ void set_biome_seed_cuda(BiomeNoiseCuda *bn, uint64_t seed) {
    XoroshiroCuda pxr;
    xset_seed_cuda(&pxr, seed);
    uint64_t xlo = xnext_long_cuda(&pxr);
    uint64_t xhi = xnext_long_cuda(&pxr);
    int n = 0;
    for (int i = 0; i < NP_MAX; i++) {
        n += init_climate_seed_cuda(&bn->climate[i], bn->oct + n, xlo, xhi, i);
    }
    bn->nptype = -1;
}

__device__ int climate_to_biome_26_2_cuda(const uint64_t np_raw[6]) {
    int64_t np[6];
    for (int i = 0; i < 6; i++) {
        np[i] = (int64_t)np_raw[i];
    }

    uint64_t best_dist = UINT64_MAX;
    int best_biome = BIOME_NONE;
    for (int i = 0; i < BIOME_PARAM_26_2_COUNT; i++) {
        const BiomeParam26_2Cuda *p = &BIOME_PARAMS_26_2[i];
        uint64_t ds = 0;
        for (int j = 0; j < 6; j++) {
            int64_t lo = (int64_t)p->v[2 * j];
            int64_t hi = (int64_t)p->v[2 * j + 1];
            int64_t d = 0;
            if (np[j] < lo) {
                d = lo - np[j];
            } else if (np[j] > hi) {
                d = np[j] - hi;
            }
            ds += (uint64_t)(d * d);
        }
        if (ds < best_dist) {
            best_dist = ds;
            best_biome = (int)p->biome;
            if (ds == 0) break;
        }
    }
    return best_biome;
}

__device__ int sample_biome_noise_cuda(const BiomeNoiseCuda *bn, int x, int y, int z, uint32_t sample_flags) {
    float t = 0, h = 0, c = 0, e = 0, d = 0, w = 0;
    double px = x, pz = z;
    if (!(sample_flags & SAMPLE_NO_SHIFT)) {
        px += sample_double_perlin_cuda(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
        pz += sample_double_perlin_cuda(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;
    }
    c = (float)sample_double_perlin_cuda(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    e = (float)sample_double_perlin_cuda(&bn->climate[NP_EROSION], px, 0, pz);
    w = (float)sample_double_perlin_cuda(&bn->climate[NP_WEIRDNESS], px, 0, pz);
    if (!(sample_flags & SAMPLE_NO_DEPTH)) {
        float np_param[] = { c, e, -3.0F * (fabsf(fabsf(w) - 0.6666667F) - 0.33333334F), w };
        float off = get_spline_cuda(bn->sp, np_param) + 0.015F;
        d = 1.0F - (y * 4) / 128.0F - 83.0F / 160.0F + off;
    }
    t = (float)sample_double_perlin_cuda(&bn->climate[NP_TEMPERATURE], px, 0, pz);
    h = (float)sample_double_perlin_cuda(&bn->climate[NP_HUMIDITY], px, 0, pz);
    uint64_t np[6];
    np[0] = (uint64_t)((int64_t)(10000.0F * t));
    np[1] = (uint64_t)((int64_t)(10000.0F * h));
    np[2] = (uint64_t)((int64_t)(10000.0F * c));
    np[3] = (uint64_t)((int64_t)(10000.0F * e));
    np[4] = (uint64_t)((int64_t)(10000.0F * d));
    np[5] = (uint64_t)((int64_t)(10000.0F * w));
    return (sample_flags & SAMPLE_NO_BIOME) ? BIOME_NONE : climate_to_biome_26_2_cuda(np);
}

__device__ int sample_biome_scaled_cuda(const BiomeNoiseCuda *bn, int scale, int x, int y, int z) {
    if (scale <= 0) scale = 4;
    int sample_scale = scale > 4 ? scale / 4 : 1;
    int mid = sample_scale / 2;
    uint32_t flags = scale > 4 ? SAMPLE_NO_SHIFT : 0;
    return sample_biome_noise_cuda(bn, x * sample_scale + mid, y, z * sample_scale + mid, flags);
}

__device__ __forceinline__ bool is_deep_ocean_cuda(int id) {
    return id == BIOME_DEEP_OCEAN || id == BIOME_DEEP_WARM_OCEAN ||
           id == BIOME_DEEP_LUKEWARM_OCEAN || id == BIOME_DEEP_COLD_OCEAN ||
           id == BIOME_DEEP_FROZEN_OCEAN;
}

__device__ __forceinline__ bool is_monument_outer_biome_cuda(int id) {
    return id == BIOME_OCEAN || id == BIOME_DEEP_OCEAN ||
           id == BIOME_RIVER || id == BIOME_FROZEN_RIVER ||
           id == BIOME_FROZEN_OCEAN || id == BIOME_DEEP_FROZEN_OCEAN ||
           id == BIOME_COLD_OCEAN || id == BIOME_DEEP_COLD_OCEAN ||
           id == BIOME_LUKEWARM_OCEAN || id == BIOME_DEEP_LUKEWARM_OCEAN ||
           id == BIOME_WARM_OCEAN || id == BIOME_DEEP_WARM_OCEAN;
}

__device__ bool monument_viable_118_cuda(const BiomeNoiseCuda *bn, int block_x, int block_z) {
    int sample_x = block_x + 8;
    int sample_z = block_z + 8;
    int id = sample_biome_scaled_cuda(bn, 4, sample_x >> 2, 36 >> 2, sample_z >> 2);
    if (!is_deep_ocean_cuda(id)) return false;
    int rad = 29;
    int x1 = (sample_x - rad) >> 2;
    int x2 = (sample_x + rad) >> 2;
    int z1 = (sample_z - rad) >> 2;
    int z2 = (sample_z + rad) >> 2;
    int y = (63 - rad) >> 2;
    for (int x = x1; x <= x2; x++) {
        for (int z = z1; z <= z2; z++) {
            int bid = sample_biome_noise_cuda(bn, x, y, z, 0);
            if (!is_monument_outer_biome_cuda(bid)) return false;
        }
    }
    return true;
}
