# v0.1.1

## 搜索流程更新

- 新增女巫小屋后处理脚本：读取筛选后的 seed CSV，扫描半径内女巫小屋，按 cubiomes 近似 surface Y 找每个 seed 的最低小屋并排序输出；支持 `--negative-y-only` 只寻找负 Y 小屋，或 `--max-y N` 限制最高 Y。
- 默认搜索目标改为：正负 60000 格四联海底神殿 -> 正负 60000 格 `17x17` 窗口最大史莱姆区块数量 -> 正负 10000 格全部主世界生物群系。
- 新增 CUDA 版四联海底神殿结构搜索，并加入 CUDA biome viability 检查。
- 新增 CUDA 版精确 `17x17` 史莱姆窗口扫描。
- 新增 CUDA 版全生物群系筛选，使用 Mojang 官方 `26.2` 正式版数据生成器报告导出的 biome 参数表。
- 默认 pipeline 不再静默回落 CPU；CPU/cubiomes 只用于显式 `--backend cpu` 复核。

## 修复

- CPU/CUDA 史莱姆区块公式精确模拟 Java 原版里的 32-bit `int` 溢出行为。
- Linux 版 CUDA helper 直接使用系统 `nvcc/gcc` 构建，并提供 Ubuntu 22.04 依赖检查脚本。

## 重要提示

- 如果使用过 v0.1.0 生成候选，请重新运行筛选。
- 本版本已切换到 Mojang 官方 `26.2` 正式版数据生成器报告。
- `scale=64` 的全生物群系阶段是快速筛选；更严格的扫描可改为 `scale=16` 或 `scale=4`。
