# Interface Spec

## prices.parquet
Columns:
- trade_date
- asset
- open
- high
- low
- close
- volume
- amount

## dataset_*.npz
Contains:
- X: shape [num_samples, 20, N]
- C: shape [num_samples, K]
- y_tail: shape [num_samples]
- state_label: shape [num_samples]

## sample outputs
- file name format: sample_{condition}_{ckpt}.npy
- generated paths shape: [num_samples, 20, N]
