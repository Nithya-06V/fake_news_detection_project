import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "linear_svm": "linear_svm.pkl",
    "random_forest": "random_forest.pkl",
    "tfidf_vectorizer": "tfidf_vectorizer.pkl",
}
MODEL_OUTPUT_DIRECTORY = "models_retrained"


def get_project_root():
    return Path(__file__).resolve().parent


def get_dataset_paths():
    root = get_project_root()
    primary_fake = root / "dataset" / "Fake.csv"
    primary_true = root / "dataset" / "True.csv"
    fallback_fake = root / "News _dataset" / "Fake.csv"
    fallback_true = root / "News _dataset" / "True.csv"

    if primary_fake.exists() and primary_true.exists():
        return primary_fake, primary_true
    if fallback_fake.exists() and fallback_true.exists():
        return fallback_fake, fallback_true
    return primary_fake, primary_true


# Detect the available text columns dynamically instead of assuming a fixed dataset schema.
def detect_text_columns(df):
    columns_map = {str(col).strip().lower(): str(col) for col in df.columns}

    if "title" in columns_map and "text" in columns_map:
        return columns_map["title"], columns_map["text"]
    if "text" in columns_map:
        return None, columns_map["text"]
    if "title" in columns_map:
        return columns_map["title"], None
    if "subject" in columns_map:
        return None, columns_map["subject"]

    available = list(df.columns)
    raise ValueError(
        "A text column required for training was not found. Available columns: "
        f"{available}. Required text column(s): title and/or text."
    )


# Clean and normalize the article text before converting it into TF-IDF features.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_article_text(title, content):
    pieces = []
    if title is not None and str(title).strip():
        pieces.append(str(title).strip())
    if content is not None and str(content).strip():
        pieces.append(str(content).strip())
    return " ".join(pieces)


def build_combined_dataset():
    fake_path, true_path = get_dataset_paths()

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Dataset files are missing. Expected Fake.csv and True.csv in {fake_path.parent} or {get_project_root() / 'dataset'}"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_title_column, fake_text_column = detect_text_columns(fake_df)
    true_title_column, true_text_column = detect_text_columns(true_df)

    fake_df["combined_text"] = fake_df.apply(
        lambda row: combine_article_text(
            row.get(fake_title_column) if fake_title_column else "",
            row.get(fake_text_column) if fake_text_column else "",
        ),
        axis=1,
    )
    true_df["combined_text"] = true_df.apply(
        lambda row: combine_article_text(
            row.get(true_title_column) if true_title_column else "",
            row.get(true_text_column) if true_text_column else "",
        ),
        axis=1,
    )

    fake_df["label"] = 0
    true_df["label"] = 1

    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df["combined_text"] = combined_df["combined_text"].apply(clean_text)
    combined_df = combined_df.dropna(subset=["combined_text"])
    combined_df = combined_df[combined_df["combined_text"].str.strip() != ""]
    combined_df = combined_df.drop_duplicates(subset=["combined_text"], keep="first")
    return combined_df, fake_df, true_df, fake_path, true_path


# Train the three required models using TF-IDF features from the cleaned dataset.
def train_models():
    combined_df, _, _, _, _ = build_combined_dataset()

    X = combined_df["combined_text"]
    y = combined_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svm": LinearSVC(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    model_results = {}
    root = get_project_root()
    model_dir = root / MODEL_OUTPUT_DIRECTORY
    model_dir.mkdir(exist_ok=True)

    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        prediction = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, zero_division=0)
        recall = recall_score(y_test, prediction, zero_division=0)
        f1 = f1_score(y_test, prediction, zero_division=0)
        confusion = confusion_matrix(y_test, prediction)
        report = classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=["FAKE", "TRUE"],
            output_dict=True,
            zero_division=0,
        )

        model_results[model_name] = {
            "model_name": model_name,
            "display_name": {
                "logistic_regression": "Logistic Regression",
                "linear_svm": "Linear SVM",
                "random_forest": "Random Forest",
            }[model_name],
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": confusion.tolist(),
            "classification_report": report,
        }

        joblib.dump(model, model_dir / MODEL_FILES[model_name])

    joblib.dump(vectorizer, model_dir / MODEL_FILES["tfidf_vectorizer"])

    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": model_results,
    }
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    try:
        from database import save_model_metrics

        metric_rows = []
        for result in model_results.values():
            metric_rows.append(
                {
                    "model_name": result["display_name"],
                    "accuracy": result["accuracy"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"],
                }
            )
        save_model_metrics(metric_rows)
    except Exception:
        pass

    return model_results


def print_training_summary(model_results):
    print("\n=== Fake News Detection Model Training Summary ===")
    for model_name, result in model_results.items():
        print(f"\nModel: {result['display_name']}")
        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall   : {result['recall']:.4f}")
        print(f"F1 Score : {result['f1_score']:.4f}")
        print(f"Confusion Matrix: {result['confusion_matrix']}")

    print("\nModel Comparison Table")
    print("-" * 80)
    print(f"{'Model':<25}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1 Score':>10}")
    for model_name, result in model_results.items():
        print(
            f"{result['display_name']:<25}{result['accuracy']:>10.4f}{result['precision']:>12.4f}{result['recall']:>10.4f}{result['f1_score']:>10.4f}"
        )
    print("-" * 80)


def main():
    try:
        fake_path, true_path = get_dataset_paths()
        print("Checking dataset files...")
        print(f"Fake.csv path: {fake_path}")
        print(f"True.csv path: {true_path}")

        if not fake_path.exists() or not true_path.exists():
            raise FileNotFoundError("Fake.csv and True.csv were not found in the expected dataset folders.")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        print(f"Fake dataset shape: {fake_df.shape}")
        print(f"True dataset shape: {true_df.shape}")
        print(f"Fake columns: {list(fake_df.columns)}")
        print(f"True columns: {list(true_df.columns)}")

        combined_df, _, _, _, _ = build_combined_dataset()
        print(f"Combined cleaned dataset shape: {combined_df.shape}")

        model_results = train_models()
        print_training_summary(model_results)
        print("\nTraining completed successfully. Models and metrics have been saved.")
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
