# Romulo DS Tools

Personal, clonable template for data science projects. The goal is not to be a public library. It is a practical starting point for plugging in a dataset, changing a few project hooks, and quickly building models, validation, visualizations, and dashboards.

## What V1 Covers

- Local files: CSV, Excel, and Parquet.
- Tabular classification and regression with scikit-learn.
- Time-series forecasting framed as supervised learning with lags and rolling windows.
- Data contracts with Pandera.
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
romds train --config conf/config.yaml
romds evaluate --config conf/config.yaml
romds dashboard --config conf/config.yaml
```

Override config values from the CLI:

```powershell
romds train --config conf/config.yaml --override data.target=price --override data.problem_type=regression
```

## Main Customization Points

Project-specific code lives in `project/`:

- `schema.py`: Pandera schema for input data.
- `preprocessing.py`: deterministic data cleaning.
- `features.py`: project-specific feature engineering.
- `models.py`: custom estimator factory.
- `visualization.py`: extra Plotly figures for reports or Dash.

Keep these hooks small and test them. The core package provides the reusable workflow; the `project/` folder carries the parts that should change per dataset.

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
