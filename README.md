# Romulo DS Tools

Personal, clonable template for data science projects. The goal is not to be a public library. It is a practical starting point for plugging in a dataset, changing a few project hooks, and quickly building models, validation, visualizations, and dashboards.

## What V1 Covers

- Local files: CSV, Excel, and Parquet.
- Tabular classification and regression with scikit-learn.
- Time-series forecasting framed as supervised learning with lags and rolling windows.
- Data contracts with Pandera.
- Standard EDA visuals after validation: box plots, correlation heatmap, and pairplot.
- Reproducible configuration with Hydra/OmegaConf-style YAML and CLI overrides.
- Experiment tracking with MLflow.
- Dataset/model orchestration with DVC.
- Generic Dash/Plotly dashboard for EDA, metrics, and prediction artifacts.
- Strong pytest suite for safe refactoring.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
romds validate --config conf/config.yaml
romds profile --config conf/config.yaml
romds train --config conf/config.yaml
romds evaluate --config conf/config.yaml
romds dashboard --config conf/config.yaml
```

After datasets are registered, the shorter workflow is:

```powershell
romds datasets list
romds doctor --dataset nursery
romds profile --dataset nursery
romds train --dataset nursery
```

## Simple First Workflow

Use this sequence when starting with a new CSV:

```powershell
# 1. Put the file under data/raw/
# Example: data/raw/my_dataset.csv

# 2. Register the dataset and generate a draft schema
romds datasets init my_dataset --data data/raw/my_dataset.csv --target target_column --active

# 3. Check paths, target column, and schema import
romds doctor --dataset my_dataset

# 4. Validate the raw data against the generated schema
romds validate --dataset my_dataset

# 5. Generate profile and initial EDA reports
romds profile --dataset my_dataset

# 6. Train a baseline model
romds train --dataset my_dataset

# 7. Recompute metrics from saved predictions
romds evaluate --dataset my_dataset
```

After step 2, review `project/schemas/my_dataset.py`. The generated schema is a starting point; tighten it as you learn the dataset. Outputs will be isolated under `reports/my_dataset_v1/` and `models/my_dataset_v1/`.

To remove generated reports for a dataset without touching `data/raw/`, use:

```powershell
# Dry run: shows what would be deleted
romds datasets delete my_dataset

# Confirm deletion
romds datasets delete my_dataset --yes
```

Override config values from the CLI:

```powershell
romds train --config conf/config.yaml --override data.target=price --override data.problem_type=regression
```

## Switching Datasets

Keep common defaults in `conf/base.yaml`, register datasets in `conf/datasets.yaml`, and create one config folder per dataset:

```text
conf/
  config.yaml
  base.yaml
  datasets.yaml
  prostate_v1/prostate_01.yaml
  nursery_v1/nursery_01.yaml
```

Dataset configs use `extends` to inherit from the base file and override only dataset-specific values:

```yaml
extends: ../base.yaml

paths:
  data: data/raw/nursery.csv
  reports_dir: reports/nursery_v1

validation:
  schema_module: project.schemas.nursery

data:
  target: "final evaluation"
```

The preferred interface is by dataset name:

```powershell
romds datasets list
romds datasets show nursery
romds datasets set-active nursery
romds validate --dataset nursery
romds profile --dataset nursery
romds train --dataset nursery
```

Run a config directly when you want an explicit experiment file:

```powershell
romds profile --config conf/prostate_v1/prostate_01.yaml
romds profile --config conf/nursery_v1/nursery_01.yaml
```

Or make `conf/config.yaml` the active dataset selector:

```yaml
extends: nursery_v1/nursery_01.yaml
```

Then all default commands keep working:

```powershell
romds train --config conf/config.yaml
```

To create a new dataset entry from a local file:

```powershell
romds datasets init my_dataset --data data/raw/my_dataset.csv --target target_column --active
romds doctor --dataset my_dataset
romds profile --dataset my_dataset
```

`datasets init` generates a dataset config and a draft schema under `project/schemas/`. Review the generated schema before treating the validation as final.

## Main Customization Points

Project-specific code lives in `project/`:

- `schema.py`: Pandera schema for input data.
- `preprocessing.py`: deterministic data cleaning.
- `features.py`: project-specific feature engineering.
- `models.py`: custom estimator factory.
- `visualization.py`: extra Plotly figures for reports or Dash.

Keep these hooks small and test them. The core package provides the reusable workflow; the `project/` folder carries the parts that should change per dataset.

## Initial EDA

`romds profile` validates the dataset when `validation.enabled=true`, writes `reports/profile.json`, and generates default HTML visualizations in `reports/eda/`.

Control the output in `conf/config.yaml`:

- `visualization.columns`: explicit columns to consider, or `null` for all.
- `visualization.exclude_columns`: extra columns to ignore.
- `visualization.include_target`: whether the target can appear as a plotted numeric feature.
- `visualization.boxplots.enabled`, `visualization.heatmap.enabled`, `visualization.pairplot.enabled`: turn each chart on or off.
- `visualization.pairplot.max_features`: limits pairplot size.
- `visualization.pairplot.color_column`: optional categorical column for pairplot color.

## Testing Philosophy

The project is expected to change often. Tests are organized around contracts rather than internal implementation details:

- Unit tests for data loading, validation, feature generation, and metrics.
- Contract tests for project hooks.
- CLI smoke tests for end-to-end behavior.
- Explicit time-series tests to prevent future leakage.
- Integration markers for DVC/MLflow-heavy checks.

Run the fast suite during refactors:

```powershell
pytest
```

Run checks before committing:

```powershell
ruff check .
ruff format --check .
pytest
```

## DVC Flow

The included `dvc.yaml` defines stages that call the same CLI commands used locally. After replacing `data/raw/example.csv` with a real dataset and updating `conf/config.yaml`, the intended flow is:

```powershell
dvc repro
```

Use DVC remotes later when datasets and model artifacts become too large for regular Git.
