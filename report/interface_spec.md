# 接口说明

## prices.parquet

字段如下：
- `trade_date`
- `asset`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`

约定：
- 按 (`trade_date`, `asset`) 排序
- `trade_date` 类型为 `datetime64[ns]`
- 数值字段会被解析为数值类型
- 在部分资产上市日期之前，默认输出允许存在跨资产不平衡情况

## dataset_*.npz

包含以下内容：
- `X`: 形状为 `[num_samples, 20, N]`
- `C`: 形状为 `[num_samples, K]`
- `y_tail`: 形状为 `[num_samples]`
- `state_label`: 形状为 `[num_samples]`
- `trade_date`: 形状为 `[num_samples]`，类型为 `datetime64[D]`
- `asset_names`: 形状为 `[N]`
- `condition_names`: 形状为 `[K]`
- `split`: 形状为 `[1]`

本仓库当前取值：
- `N = 5` 个资产
- `K = 4` 个条件变量，分别为 `cumret_5d`、`vol_20d`、`amount_change_5d`、`high_vol`
- 窗口长度 `window length = 20`
- 文件包括：
  - `data/processed/dataset_train.npz`
  - `data/processed/dataset_valid.npz`
  - `data/processed/dataset_test.npz`

相关文件：
- `data/processed/features.parquet`
- `data/processed/tail_stats.json`
- `outputs/tables/b_dataset_composition.csv`
- `outputs/tables/b_feature_descriptive_stats.csv`

## 采样输出

- 文件命名格式：`sample_{condition}_{ckpt}.npy`
- 生成路径张量形状：`[num_samples, 20, N]`
- 数据类型：`float32`

附加采样产物：
- `sample_{condition}_{ckpt}_price.npy`
  - 显式包含基准时点 `t0=1`
  - 形状：`[num_samples, 21, N]`
  - 类型：`float32`
- `sample_{condition}_{ckpt}_traj.npy`（当 `save_trajectory=true` 时生成）
  - 稀疏保存的反向扩散轨迹
  - 类型：`float32`
- `sample_{condition}_{ckpt}_traj_steps.npy`（当 `save_trajectory=true` 时生成）
  - 与 `traj` 第一维对齐的整数时间步索引
  - 顺序为从 `T` 到 `0` 的反向采样顺序
  - 类型：`int32`

说明：
- `sample_{condition}_{ckpt}.npy` 保存的是 20 步对数收益路径。
- `_price.npy` 由对数收益通过累加后再指数化得到。
- 轨迹保存使用固定 `save_interval=20`，并始终保留 `T` 和 `0` 两个端点。

## 校准输出

- `outputs/calibration/calibration_summary.csv`
  - 列包括：
    - `seed`
    - `tail_weight`
    - `checkpoint_used`
    - `is_baseline`
    - `artifacts_scope`
    - `mean_wasserstein`
    - `mean_ks_stat`
    - `mean_abs_es_gap`
    - `mean_abs_var_gap`
    - `run_dir`
    - `rank`
- `outputs/calibration/seed_{seed}_tailw_{tail_weight}/`
  - 表示一次“仅评估用途”的校准快照目录
  - 其中包含复制后的 `checkpoints`、`logs`、`samples`、`tables`、`figures` 以及 `run_meta.json`

约定：
- 校准网格固定为 `seed in {42, 52, 62}` 与 `tail_weight in {1.0, 3.0, 5.0}`
- 校准阶段的生成与评估仅使用 `best` checkpoint
- 校准调用方式为 `sample.py --save-trajectory false`
- `artifacts_scope` 固定为 `evaluation_only`
- 排名优先级为：`mean_wasserstein` 升序，然后 `mean_abs_es_gap`，再然后 `mean_ks_stat`

## D 模块输出

### 评估表

- `outputs/tables/d_real_risk_metrics.csv`
  - 列包括：
    - `horizon`
    - `alpha`
    - `baseline_scope`（`overall_test`）
    - `count`
    - `mean`
    - `std`
    - `var_5pct`
    - `es_5pct`
    - `var_alpha`
    - `es_alpha`
    - `worst`
- `outputs/tables/d_generated_risk_metrics.csv`
  - 列包括：
    - `condition`
    - `checkpoint`
    - `horizon`
    - `alpha`
    - `count`
    - `mean`
    - `std`
    - `var_5pct`
    - `es_5pct`
    - `var_alpha`
    - `es_alpha`
    - `worst`
    - `source_file`
- `outputs/tables/d_distribution_compare.csv`
  - 列包括：
    - `condition`
    - `checkpoint`
    - `horizon`
    - `ks_stat`
    - `ks_pvalue`
    - `spearman_hist`
    - `wasserstein_distance`

### 归因表

- `outputs/tables/d_factor_sensitivity.csv`
  - 当前列包括：
    - `checkpoint`
    - `base_condition`
    - `factor`
    - `factor_value`
    - `horizon`
    - `alpha`
    - `count`
    - `mean`
    - `std`
    - `var_5pct`
    - `es_5pct`
    - `var_alpha`
    - `es_alpha`
    - `worst`
    - `delta_var_5pct`
    - `delta_es_5pct`
- `outputs/tables/d_factor_sensitivity_meta.json`
  - 字段包括：
    - `checkpoint`
    - `base_condition`
    - `n_samples`
    - `factor_values`
    - `high_vol_values`
    - `timesteps`
    - `seed`

### D 模块图像

- `outputs/figures/d_return_dist_last_day.png`
- `outputs/figures/d_left_tail_compare_last_day.png`
- `outputs/figures/d_es_by_condition_last_day.png`
- `outputs/figures/d_return_dist_cum_20d.png`
- `outputs/figures/d_left_tail_compare_cum_20d.png`
- `outputs/figures/d_es_by_condition_cum_20d.png`
- `outputs/figures/d_factor_sensitivity_es_last_day.png`
- `outputs/figures/d_factor_sensitivity_es_cum_20d.png`

约定：
- 评估时间尺度使用 `last_day` 与 `cum_20d`。
- `evaluate.py` 支持 `--checkpoint-filter latest|best|all`，默认值为 `latest`。
- 归因结果同时包含 baseline 记录与各单因子扰动记录。
