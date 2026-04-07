# 数据字典

## `data/processed/prices.parquet`

| 列名 | 类型 | 含义 |
|---|---|---|
| `trade_date` | datetime64[ns] | 资产行情对应的交易日期 |
| `asset` | string | 资产标识符，例如 `hs300_etf` |
| `open` | float | 当日开盘价 |
| `high` | float | 当日最高价 |
| `low` | float | 当日最低价 |
| `close` | float | 当日收盘价 |
| `volume` | numeric | 成交量 |
| `amount` | float | 成交额 / 换手金额 |

说明：
- `prices.parquet` 按 (`trade_date`, `asset`) 排序。
- 默认预处理保留各资产完整历史；如有需要，可额外启用公共日期对齐。

## `data/processed/features.parquet`

| 列组 | 含义 |
|---|---|
| `portfolio_return` | 等权组合对数收益率 |
| `tail_loss` | `-portfolio_return` |
| `cumret_5d` | 5 日累计组合收益 |
| `vol_20d` | 20 日滚动组合波动率 |
| `amount_change_5d` | 5 日组合成交额对数变化 |
| `high_vol` | 高波动状态二值标签，依据训练集波动率中位数划分 |
| `y_tail` | 尾部事件二值标签，定义为 `tail_loss` >= 训练集 95% 分位数 |
| `ret_*` | 各资产的对数收益率列 |
| `split` | 时间划分标签，取值为 `train` / `valid` / `test` |

## `data/processed/dataset_*.npz`

| 键名 | 形状 | 含义 |
|---|---|---|
| `X` | `[num_samples, 20, N]` | 对数收益路径张量 |
| `C` | `[num_samples, 4]` | 条件向量，包含 `cumret_5d`、`vol_20d`、`amount_change_5d`、`high_vol` |
| `y_tail` | `[num_samples]` | 尾部事件标签 |
| `state_label` | `[num_samples]` | 高波动状态标签 |
| `trade_date` | `[num_samples]` | 每个样本窗口的结束日期 |
| `asset_names` | `[N]` | 资产通道名称 |
| `condition_names` | `[4]` | 条件变量名称 |
| `split` | `[1]` | 当前数据集所属划分 |
