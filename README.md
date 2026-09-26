# deep-learning-fake-image-detection

## Reproducible Strict Pipeline

This project supports a reproducible stratified train/valid/test workflow.

### 1) Build deterministic split

```bash
python app/build_stratified_splits.py \
	--dataset-root dataset \
	--source-mode existing-splits \
	--output-dir dataset/strict_split \
	--seed 42 \
	--valid-ratio 0.15 \
	--test-ratio 0.10 \
	--link-mode hardlink \
	--overwrite
```

This creates:

- dataset/strict_split/train/fake, dataset/strict_split/train/real
- dataset/strict_split/valid/fake, dataset/strict_split/valid/real
- dataset/strict_split/test/fake, dataset/strict_split/test/real
- dataset/strict_split/manifests/{train.csv,valid.csv,test.csv,metadata.json}

### 2) Train on strict split

```bash
python app/train.py \
	--dataset-root dataset/strict_split \
	--model-path app/models/best_model.keras \
	--seed 42 \
	--img-size 300 \
	--batch-size 16 \
	--epochs-phase1 10 \
	--epochs-phase2 6
```

Training writes run metadata to app/models/last_train_run.json.

### 3) Tune threshold on validation only

```bash
python app/tune_threshold.py \
	--dataset-root dataset/strict_split \
	--model-path app/models/best_model.keras
```

### 4) Strict test evaluation

```bash
python app/eval_valid.py \
	--dataset-root dataset/strict_split \
	--split test \
	--model-path app/models/best_model.keras \
	--progress-every 100
```

### One-command pipeline

```bash
python app/run_strict_pipeline.py --overwrite-split
```

### Optional utilities (not required for core pipeline)

The following scripts were moved to keep `app/` focused on core pipeline modules:

- `app/optional_tools/sanity_check.py`
- `app/optional_tools/mine_hard_examples.py`
- `app/optional_tools/simple_eval.py`

Generated logs are stored under:

- `app/logs/`

Runtime web uploads are stored under:

- `app/static/uploads/`