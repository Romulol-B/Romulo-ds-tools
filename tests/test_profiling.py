from __future__ import annotations

from romulo_ds_tools.profiling import run_profile, summarize_dataframe


def test_summarize_dataframe_reports_shape_dtypes_and_missing_values(classification_frame):
    summary = summarize_dataframe(classification_frame)

    assert summary["rows"] == len(classification_frame)
    assert summary["columns"] == len(classification_frame.columns)
    assert summary["missing"]["feature_num"] == 0
    assert "feature_num" in summary["numeric_summary"]


def test_summarize_dataframe_supports_all_categorical_data():
    import pandas as pd

    summary = summarize_dataframe(pd.DataFrame({"a": ["x", "y"], "b": ["m", "n"]}))

    assert summary["rows"] == 2
    assert summary["numeric_summary"] == {}


def test_run_profile_writes_json_and_missingness_plot(
    tmp_path, classification_frame, config_factory
):
    data_path = tmp_path / "train.csv"
    classification_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)

    summary = run_profile(cfg)

    assert summary["rows"] == len(classification_frame)
    assert (tmp_path / "reports" / "profile.json").exists()
    assert (tmp_path / "reports" / "missingness.html").exists()


def test_run_profile_validates_then_writes_standard_eda(tmp_path, prostate_frame, config_factory):
    data_path = tmp_path / "prostate.csv"
    prostate_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)
    cfg.validation.enabled = True
    cfg.data.target = "diagnosis_result"
    cfg.data.id_columns = ["id"]
    cfg.visualization = {
        "enabled": True,
        "output_dir": str(tmp_path / "reports" / "eda"),
        "columns": None,
        "exclude_columns": [],
        "include_target": False,
        "boxplots": {"enabled": True},
        "heatmap": {"enabled": True},
        "pairplot": {
            "enabled": True,
            "max_features": 3,
            "color_column": "diagnosis_result",
        },
    }

    run_profile(cfg)

    assert (tmp_path / "reports" / "eda" / "boxplots.html").exists()
    assert (tmp_path / "reports" / "eda" / "correlation_heatmap.html").exists()
    assert (tmp_path / "reports" / "eda" / "pairplot.html").exists()
    assert (tmp_path / "reports" / "eda" / "manifest.json").exists()
