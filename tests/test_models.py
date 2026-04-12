from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models import build_model, get_param_grid


def test_get_param_grid_contains_expected_models():
    grid = get_param_grid()

    assert "LogisticRegression" in grid
    assert "RandomForest" in grid
    assert isinstance(grid["LogisticRegression"], list)
    assert isinstance(grid["RandomForest"], list)


def test_build_model_logistic_regression():
    model = build_model("LogisticRegression", {"C": 1.0, "max_iter": 200})

    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0
    assert model.max_iter == 200


def test_build_model_random_forest():
    model = build_model("RandomForest", {"n_estimators": 50})

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50


def test_build_model_unsupported_raises_value_error():
    try:
        build_model("SVM", {})
        assert False, "Expected ValueError for unsupported model"
    except ValueError as exc:
        assert "Unsupported model" in str(exc)
