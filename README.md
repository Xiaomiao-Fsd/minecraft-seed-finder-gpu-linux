# Minecraft Seed Finder GPU - Linux

Ubuntu 22.04 可用的 Minecraft Java `26.2` 种子筛选工具，默认 pipeline 是 GPU-only：

1. CUDA 搜索原点正负 60000 格内的四联海底神殿。四个神殿的完整 footprint 必须能被一个 chunk 对齐的 `17x17` 区块正方形完整包含，并默认做海洋 biome viability 检查。
2. CUDA 对候选 seed 精确扫描正负 60000 格内所有完整 `17x17` 窗口，输出最大史莱姆区块数量。
3. CUDA 在原点正负 10000 格内扫描全部主世界生物群系。

CUDA biome noise 参考 Mojang 官方 Java `26.2` 正式版数据生成器报告，参数表包含 55 个主世界生物群系。

## Ubuntu 22.04 依赖

必须有 NVIDIA 驱动和 CUDA Toolkit 的 `nvcc`：

```bash
nvidia-smi
nvcc --version
```

基础编译依赖：

```bash
sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-pip bash coreutils
```

也可以运行辅助检查脚本：

```bash
bash scripts/install_ubuntu22_deps.sh
```

这个脚本会安装通用编译依赖并检查 `nvidia-smi`、`nvcc`、`gcc`、`python3`。如果机器没有 CUDA Toolkit，请按 NVIDIA 官方 Ubuntu 22.04 CUDA 安装说明安装匹配驱动的 Toolkit。

## 快速开始

```bash
python3 seed_finder.py build --target all --backend cuda
python3 seed_finder.py pipeline --count 1000000 --backend cuda
```

长期/可恢复任务：

```bash
python3 seed_cluster_runner.py start --count 1000000000 --backend cuda --gpus 0
```

或：

```bash
MC_SEED_BACKEND=cuda MC_SEED_GPUS=0 bash run_seed_cluster.sh 1b
```

多 GPU：

```bash
MC_SEED_BACKEND=cuda MC_SEED_GPUS=0,1,2,3 MC_SEED_CHUNK_SIZE=10000000 bash run_seed_cluster.sh 10b
```

## 输出

默认输出目录是 `runs/default`，cluster runner 会创建 `runs/cluster_YYYYMMDD_HHMMSS/`。

关键 CSV：

- `candidates_quad_monuments.csv`：四联海底神殿候选。
- `candidates_quad_monuments_slime_windows.csv`：追加 `17x17` 最大史莱姆区块数量。
- `results_all_biomes.csv`：追加全生物群系筛选结果。

## 常用命令

只跑第一阶段：

```bash
python3 seed_finder.py prefilter --count 1000000 --backend cuda
```

对已有候选计算史莱姆窗口：

```bash
python3 seed_finder.py slime-window --candidates runs/default/candidates_quad_monuments.csv --backend cuda
```

对已有候选做全生物群系 CUDA 筛选：

```bash
python3 seed_finder.py all-biomes --candidates runs/default/candidates_quad_monuments_slime_windows.csv --backend cuda
```

对筛选后的候选搜索近似 Y 最低的女巫小屋：

```bash
# 如果 cubiomes 不在默认相对路径，先设置 CUBIOMES_DIR=/path/to/cubiomes
scripts/find_lowest_witch_huts.sh \
  --backend cuda \
  --in runs/default/results_all_biomes.csv \
  --out runs/default/witch_huts_negative_y.csv \
  --whole-world \
  --negative-y-only \
  --top 50
```

如果只想限制到某个高度，也可以用：

```bash
scripts/find_lowest_witch_huts.sh \
  --backend cuda \
  --in runs/default/results_all_biomes.csv \
  --out runs/default/witch_huts_y40_or_lower.csv \
  --whole-world \
  --max-y 40
```

这个后处理会读取输入 CSV 第一列 seed，输出每个 seed 在搜索范围内近似 Y 最低的女巫小屋，并整体按 `hut_y_approx` 从低到高排序。默认走 CUDA 后端；加 `--backend cpu` 才回退到 cubiomes CPU 版。加 `--whole-world` 时按 Minecraft 世界边界完整方形区域 `±29999984` 搜索；不加时可用 `--radius-blocks N` 指定原点半径。加 `--negative-y-only` 时只保留 `hut_y_approx < 0` 的小屋；加 `--max-y N` 时只保留 `hut_y_approx <= N` 的小屋。女巫小屋 X/Z 来自结构公式；Y 在 CUDA 版里用 26.2 biome noise 的 depth 参数近似，适合快速排序/定位，最终精确方块级高度建议进游戏或 Chunkbase 复核。

注意：`--whole-world` 是精确覆盖世界边界，但每个 seed 约需扫描 `117190 x 117190 ≈ 1.37e10` 个 Swamp Hut region，会非常慢，建议只在候选 seed 数量很少、可以长期后台运行时使用。

CPU/cubiomes 复核必须显式指定：

```bash
python3 seed_finder.py all-biomes --candidates runs/default/candidates_quad_monuments_slime_windows.csv --backend cpu
```

## 关键参数

见 `config.example.json`：

- `prefilter.search_radius_blocks`: 四联神殿搜索半径，默认 60000。
- `prefilter.cluster_window_chunks`: 神殿聚类窗口，默认 17。
- `prefilter.check_biome_viability`: 是否做神殿海洋 biome viability，默认 true。
- `postfilters.slime_window_density.window_chunks`: 史莱姆窗口大小，默认 17。
- `postfilters.slime_window_density.min_slime_chunks`: 史莱姆窗口阈值，默认 0 表示只记录不丢弃。
- `postfilters.all_biomes_within_origin_radius.radius_blocks`: 全生物群系半径，默认 10000。
- `postfilters.all_biomes_within_origin_radius.scale`: biome 扫描尺度，默认 64。
- `scripts/find_lowest_witch_huts.sh` 默认使用 CUDA 后端；`--backend cpu` 仅用于 cubiomes 复核。

## 当前限制

- Linux 版默认使用系统里的 `gcc/g++/nvcc`，不包含 Windows `.exe/.lib/.exp` 编译产物。
- 默认 pipeline 不会静默回落 CPU；`auto` 找不到 CUDA 会报错。
- `scale=64` 是快速筛选；更严格可改 `scale=16` 或 `scale=4`，但会显著变慢。

## 打包

```bash
bash scripts/package_release.sh v0.1.1
```

第三方声明见 `THIRD_PARTY_NOTICES.md`。
