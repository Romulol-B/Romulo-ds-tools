from __future__ import annotations

from romulo_ds_tools.modeling import build_estimator, build_model_pipeline, prepare_data


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
