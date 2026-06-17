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
  --in runs/default/results_all_biomes.csv \
  --out runs/default/witch_huts_lowest.csv \
  --radius-blocks 60000 \
  --top 50
```

这个后处理会读取输入 CSV 第一列 seed，输出每个 seed 在搜索半径内近似 Y 最低的女巫小屋，并整体按 `hut_y_approx` 从低到高排序。女巫小屋 X/Z 来自 cubiomes 结构公式；Y 使用 cubiomes `mapApproxHeight()` 在小屋 chunk 中心估算，适合快速排序/定位，最终精确方块级高度建议进游戏或 Chunkbase 复核。

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

## 当前限制

- Linux 版默认使用系统里的 `gcc/g++/nvcc`，不包含 Windows `.exe/.lib/.exp` 编译产物。
- 默认 pipeline 不会静默回落 CPU；`auto` 找不到 CUDA 会报错。
- `scale=64` 是快速筛选；更严格可改 `scale=16` 或 `scale=4`，但会显著变慢。

## 打包

```bash
bash scripts/package_release.sh v0.1.1
```

第三方声明见 `THIRD_PARTY_NOTICES.md`。
