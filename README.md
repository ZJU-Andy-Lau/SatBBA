# SatBBA

生产级卫星影像区域网平差（Bundle Block Adjustment, BBA）工程。

SatBBA 支持从多景卫星影像与 RPC 出发，完成：
1. pairwise 匹配（可插拔 matcher）
2. 多视 tracks 构建与清洗
3. tie point 初始三维重建
4. 固定参考影像 + 非参考影像像空间 affine 修正 + tie point 3D 的稀疏 BA
5. 报告、可视化与可选 refined RPC 导出

---

## 1. 系统目标

- 在无 GCP / 无外部 DEM 条件下获得区域网内部几何一致性。
- 输出每景 affine、tie point 高程与完整误差统计。
- 保持模块化、可恢复（checkpoint/resume）、可复现（配置驱动）。

## 2. 数学模型（简述）

对观测 \((i,j)\)：
\[
\mathbf{u}^{pred}_{ij}=\Pi^{rpc}_i(\mathbf{X}_j)+\Delta_i(\mathbf{u}^{rpc}_{ij})
\]

其中：
- \(\Pi^{rpc}_i\)：原始 RPC 投影
- \(\Delta_i\)：像空间 6 参数仿射修正（参考影像固定为 0）

优化目标（鲁棒 + 弱约束）：
\[
\min \sum \rho(\|e_{ij}\|^2)+\lambda_a\sum_i\|\theta_i\|^2+\lambda_p\sum_j\|X_j-X_j^{init}\|^2
\]

## 3. 架构概览

- `satbba/matching`: matcher 插件抽象、注册、SIFT/LoFTR
- `satbba/core`: 各阶段命令编排
- `satbba/tracking`: tracks 合并与过滤
- `satbba/triangulation`: 初始 3D 点构建与质量筛选
- `satbba/ba`: BA 数据加载、状态、残差、求解器、checkpoint、RPC refit
- `satbba/reporting`: summary 与 plots
- `satbba/io`: 影像/RPC 读取与 batch RPC projection

## 4. 安装

### pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda env create -f env.yaml
conda activate satbba
pip install -e .
```


### 无需安装模式（开发机快速运行）
```bash
python main.py run-all --config configs/config.yaml
python main.py bundle-adjust --config configs/config.yaml
```

## 5. 输入数据要求

- 影像：`*.tif`
- RPC：GeoTIFF 内嵌 RPC 或 sidecar (`.RPB/.xml/.txt/.json`)

## 6. 配置

参考 `configs/config.yaml`。
关键段：
- `matching`（matcher 类型、过滤阈值）
- `tracks`（min_views 等）
- `triangulation`
- `ba`（stages/loss/reg/outlier）
- `optimization`
- `rpc_refit`
- `runtime`（checkpoint_interval/resume/force_recompute）
- `output`（plots/report/rpc 导出）

## 7. CLI 用法

```bash
sat_bba list-matchers
sat_bba match --config configs/config.yaml
sat_bba build-tracks --config configs/config.yaml
sat_bba triangulate-init --config configs/config.yaml
sat_bba bundle-adjust --config configs/config.yaml
sat_bba run-all --config configs/config.yaml
```

恢复 BA：
```bash
sat_bba bundle-adjust --config configs/config.yaml
# 在 config.runtime.resume 指向 checkpoint 文件
```

## 8. 典型工作流

```bash
sat_bba run-all --config configs/config.yaml
```

输出目录（默认 `outputs/`）包含：
- `matches/`, `pairs.json`
- `tracks/tracks.jsonl`, `tracks/track_summary.json`
- `ba_input/*.csv`
- `ba_results/affine_params.json`, `points_optimized.json`, `ba_report.json`
- `rpc_refined/*.rpc`（启用时）
- `plots/*.png`（启用时）
- `report.json`

## 9. Matcher 插件机制

- 通过 `BaseMatcher` + `register_matcher` 注册。
- 在配置中用 `matching.matcher.type` 切换 matcher。

### LoFTR 权重本地加载

- 必须显式配置 `weights_path`。
- 路径不存在会抛出明确错误。
- 不允许自动下载权重。

## 10. 性能建议

- 开启 `optimization.use_numba=true`（若环境支持）
- 合理设置 `optimization.batch_size`
- 大区域优先先跑 `match/build-tracks/triangulate-init` 并缓存，再跑 BA

## 11. 常见问题

1. **找不到 LoFTR 权重**：检查 `matching.matcher.weights_path` 是否存在。  
2. **依赖缺失导致测试 skip**：安装 `numpy/scipy/opencv/torch` 后重跑。  
3. **结果已存在未重算**：设置 `runtime.force_recompute=true`。

## 12. 测试

```bash
pytest
```

## 13. 开发扩展

- 新 matcher：实现 `BaseMatcher` 并注册。
- 新 BA 约束：在 `ba/residuals.py` 扩展项，并同步 `jac_sparsity`。
- 新导出格式：在 `reporting/` 与 `ba/solver.py` 导出步骤追加。

## 14. 局限性

- 当前 refined RPC 为“RPC+affine 校正”的紧凑拟合导出（sidecar JSON `.rpc`），非完整供应商 RPC 全参数反演。
- 极大规模数据集仍建议分块处理与外部调度。

## 15. 后续方向

- 完整 corrected RPC 全系数反演
- 更强的并行残差评估
- 更丰富的几何一致性质量指标与可视化
