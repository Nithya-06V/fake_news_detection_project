import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for

from config import SECRET_KEY
from database import (
    fetch_model_metrics,
    fetch_prediction_history,
    initialize_database,
    insert_prediction_record,
    validate_login,
)
from train_model import (
    MODEL_OUTPUT_DIRECTORY,
    build_combined_dataset,
    clean_text,
    combine_article_text,
    get_dataset_paths,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
}

MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "linear_svm": "linear_svm.pkl",
    "random_forest": "random_forest.pkl",
    "tfidf_vectorizer": "tfidf_vectorizer.pkl",
}


def login_required(view_function):
    from functools import wraps

    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapper


# Application utilities used by the dashboard, prediction pages, and model loading.
def get_project_root():
    return Path(__file__).resolve().parent


def get_model_files():
    model_dir = get_project_root() / MODEL_OUTPUT_DIRECTORY
    return {name: model_dir / filename for name, filename in MODEL_FILES.items()}


def load_model_artifacts():
    files = get_model_files()
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained model files are missing. Run 'python train_model.py' before using the web app."
        )

    vectorizer = joblib.load(files["tfidf_vectorizer"])
    models = {name: joblib.load(path) for name, path in files.items() if name != "tfidf_vectorizer"}
    return vectorizer, models


def load_saved_metrics():
    metrics_path = get_project_root() / MODEL_OUTPUT_DIRECTORY / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            results = data.get("results", {})
            all_metrics = []
            for key, value in results.items():
                all_metrics.append(
                    {
                        "model_name": value["display_name"],
                        "accuracy": value.get("accuracy", 0),
                        "precision": value.get("precision", 0),
                        "recall": value.get("recall", 0),
                        "f1_score": value.get("f1_score", 0),
                    }
                )
            return all_metrics

    try:
        metrics_by_model = fetch_model_metrics()
        rows = []
        for key, value in metrics_by_model.items():
            rows.append(
                {
                    "model_name": value["model_name"],
                    "accuracy": float(value.get("accuracy", 0)),
                    "precision": float(value.get("precision", 0)),
                    "recall": float(value.get("recall", 0)),
                    "f1_score": float(value.get("f1_score", 0)),
                }
            )
        return rows
    except Exception:
        return []


def get_dataset_summary():
    try:
        fake_path, true_path = get_dataset_paths()
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
    except Exception:
        return {
            "fake_records": 0,
            "true_records": 0,
            "total_records": 0,
            "fake_columns": [],
            "true_columns": [],
            "combined_columns": [],
            "fake_shape": "0x0",
            "true_shape": "0x0",
            "combined_shape": "0x0",
            "source": "Not available",
        }

    combined_df = pd.concat([fake_df.assign(label=0), true_df.assign(label=1)], ignore_index=True)
    return {
        "fake_records": len(fake_df),
        "true_records": len(true_df),
        "total_records": len(combined_df),
        "fake_columns": list(fake_df.columns),
        "true_columns": list(true_df.columns),
        "combined_columns": list(combined_df.columns),
        "fake_shape": fake_df.shape,
        "true_shape": true_df.shape,
        "combined_shape": combined_df.shape,
        "source": "Fake.csv and True.csv",
        "fake_missing_values": fake_df.isnull().sum().to_dict(),
        "true_missing_values": true_df.isnull().sum().to_dict(),
        "combined_missing_values": combined_df.isnull().sum().to_dict(),
        "duplicate_records": int(combined_df.duplicated().sum()),
        "fake_sample": fake_df.head(5).to_dict(orient="records"),
        "true_sample": true_df.head(5).to_dict(orient="records"),
        "combined_sample": combined_df.head(5).to_dict(orient="records"),
    }


def model_metrics_chart_data():
    metrics = load_saved_metrics()
    if not metrics:
        return {"labels": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    ordered_names = ["Logistic Regression", "Linear SVM", "Random Forest"]
    ordered_results = []
    for name in ordered_names:
        for metric in metrics:
            if metric["model_name"] == name:
                ordered_results.append(metric)
                break

    return {
        "labels": [item["model_name"] for item in ordered_results],
        "accuracy": [float(item["accuracy"]) for item in ordered_results],
        "precision": [float(item["precision"]) for item in ordered_results],
        "recall": [float(item["recall"]) for item in ordered_results],
        "f1": [float(item["f1_score"]) for item in ordered_results],
    }


def best_accuracy_value():
    metrics = load_saved_metrics()
    if not metrics:
        return 0.0
    return max(float(item["accuracy"]) for item in metrics)


def normalize_prediction_value(prediction_value):
    if prediction_value == 0:
        return 0, "FAKE NEWS"
    if prediction_value == 1:
        return 1, "TRUE NEWS"

    normalized_value = str(prediction_value).strip().upper()
    if normalized_value in {"0", "FAKE", "FAKE NEWS"}:
        return 0, "FAKE NEWS"
    if normalized_value in {"1", "TRUE", "TRUE NEWS"}:
        return 1, "TRUE NEWS"

    raise ValueError(f"Model returned an unsupported prediction label: {prediction_value!r}")


def build_prediction_object(title, content, model_key, prediction_value, confidence_value=None):
    prediction_value, prediction_label = normalize_prediction_value(prediction_value)
    return {
        "news_title": title or "Untitled News",
        "news_content": content,
        "selected_model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "prediction": prediction_label,
        "prediction_value": prediction_value,
        "confidence": confidence_value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "article_preview": (content[:140] + "...") if len(content) > 140 else content,
    }


# Apply the same preprocessing flow to new article text before making a prediction.
def predict_news(title, content, model_key):
    if model_key not in MODEL_DISPLAY_NAMES:
        raise ValueError(f"Unsupported model: {model_key}")

    vectorizer, models = load_model_artifacts()
    combined_text = combine_article_text(title, content)
    cleaned_text = clean_text(combined_text)
    input_vector = vectorizer.transform([cleaned_text])

    model = models[model_key]
    prediction_value = model.predict(input_vector)[0]
    confidence = None

    if model_key != "linear_svm" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_vector)[0]
        confidence = float(max(probabilities)) * 100

    return build_prediction_object(title, content, model_key, prediction_value, confidence)


@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == "admin" and password == "admin123":
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password. Please try again.", "danger")
        return render_template("login.html")

    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    dataset_summary = get_dataset_summary()
    metrics = load_saved_metrics()
    best_accuracy = best_accuracy_value()
    return render_template(
        "dashboard.html",
        dataset_summary=dataset_summary,
        metrics=metrics,
        best_accuracy=best_accuracy,
        chart_data=model_metrics_chart_data(),
        username=session.get("username", "admin"),
    )


@app.route("/dataset")
@login_required
def dataset_info():
    dataset_summary = get_dataset_summary()
    return render_template("dataset.html", dataset_summary=dataset_summary)


@app.route("/detection", methods=["GET"])
@login_required
def detection():
    return render_template("detection.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    title = request.form.get("news_title", "").strip()
    content = request.form.get("news_content", "").strip()
    model_key = request.form.get("model_name", "logistic_regression")

    if not title and not content:
        flash("Please provide a news title or article content before detection.", "danger")
        return redirect(url_for("detection"))

    if not content:
        flash("Please provide article content for prediction.", "danger")
        return redirect(url_for("detection"))

    if model_key not in MODEL_DISPLAY_NAMES:
        flash("Selected model is invalid. Please choose a valid model.", "danger")
        return redirect(url_for("detection"))

    try:
        result = predict_news(title, content, model_key)
    except FileNotFoundError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("detection"))
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "danger")
        return redirect(url_for("detection"))

    session["last_prediction"] = result
    try:
        insert_prediction_record(
            result["news_title"],
            result["news_content"],
            result["selected_model"],
            result["prediction"],
        )
    except Exception:
        app.logger.exception("Could not save the prediction to MySQL history.")
        flash("Prediction completed, but it could not be saved to history.", "warning")
    return redirect(url_for("result"))


@app.route("/result")
@login_required
def result():
    result_data = session.get("last_prediction")
    if not result_data:
        flash("No prediction result is available yet.", "warning")
        return redirect(url_for("detection"))
    return render_template("result.html", result_data=result_data)


@app.route("/history")
@login_required
def history():
    page = request.args.get("page", default=1, type=int)
    per_page = 10
    records, total_rows = fetch_prediction_history(page=page, per_page=per_page)
    total_pages = max(1, (total_rows + per_page - 1) // per_page) if total_rows else 1
    return render_template(
        "history.html",
        records=records,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_rows=total_rows,
    )


@app.route("/model-performance")
@login_required
def model_performance():
    metrics = load_saved_metrics()
    chart_data = model_metrics_chart_data()
    return render_template(
        "model_performance.html",
        metrics=metrics,
        chart_data=chart_data,
        evaluation_message="Evaluation metrics are calculated on the held-out test dataset.",
    )


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


try:
    initialize_database()
    app.config["DB_AVAILABLE"] = True
except Exception as exc:
    app.config["DB_AVAILABLE"] = False
    app.config["DB_ERROR"] = str(exc)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
