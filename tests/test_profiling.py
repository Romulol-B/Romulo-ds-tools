from __future__ import annotations

from romulo_ds_tools.profiling import run_profile, summarize_dataframe


def test_summarize_dataframe_reports_shape_dtypes_and_missing_values(classification_frame):
    summary = summarize_dataframe(classification_frame)

    assert summary["rows"] == len(classification_frame)
    assert summary["columns"] == len(classification_frame.columns)
    assert summary["missing"]["feature_num"] == 0
    assert "feature_num" in summary["numeric_summary"]


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
