from __future__ import annotations

import pytest

from romulo_ds_tools.modeling import (
    build_estimator,
    build_model_pipeline,
    build_preprocessor,
    prepare_data,
)


def test_prepare_data_uses_stratified_random_split_when_possible(classification_frame):
    prepared = prepare_data(
        classification_frame,
        target_column="target",
        problem_type="classification",
        test_size=0.25,
        random_state=42,
        id_columns=["row_id"],
    )

    assert prepared.x_train.shape[0] == 6
    assert prepared.x_test.shape[0] == 2
    assert "row_id" not in prepared.feature_columns


def test_prepare_data_uses_temporal_holdout(time_series_frame):
    prepared = prepare_data(
        time_series_frame,
        target_column="target",
        problem_type="regression",
        test_size=0.25,
        random_state=42,
        time_column="date",
    )

    assert prepared.y_train.tolist() == [10.0, 12.0, 13.0, 15.0, 18.0, 21.0]
    assert prepared.y_test.tolist() == [22.0, 25.0]


def test_sklearn_pipeline_fits_mixed_numeric_and_categorical_features(classification_frame):
    prepared = prepare_data(
        classification_frame,
        target_column="target",
        problem_type="classification",
        test_size=0.25,
        random_state=42,
        id_columns=["row_id"],
    )
    estimator = build_estimator(
        "classification",
        "random_forest_classifier",
        {"n_estimators": 5, "random_state": 42},
    )
    model = build_model_pipeline(estimator)

    model.fit(prepared.x_train, prepared.y_train)
    predictions = model.predict(prepared.x_test)

    assert len(predictions) == len(prepared.y_test)


def test_custom_column_transformer_excludes_column_from_default_selectors(
    classification_frame,
    monkeypatch,
):
    import project.preprocessing as project_preprocessing

    def get_column_transformers(cfg):
        return [("feature_num_raw", "passthrough", ["feature_num"])]

    monkeypatch.setattr(project_preprocessing, "get_column_transformers", get_column_transformers)
    preprocessor = build_preprocessor(cfg={})

    transformed = preprocessor.fit_transform(classification_frame[["feature_num", "feature_cat"]])

    assert transformed.shape[1] == 3


def test_custom_column_transformer_requires_explicit_string_columns(monkeypatch):
    import project.preprocessing as project_preprocessing

    def get_column_transformers(cfg):
        return [("bad", "passthrough", slice(None))]

    monkeypatch.setattr(project_preprocessing, "get_column_transformers", get_column_transformers)

    with pytest.raises(TypeError, match="explicit string column names"):
        build_preprocessor(cfg={})
