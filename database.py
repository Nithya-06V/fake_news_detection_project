import mysql.connector
from mysql.connector import Error

from config import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER


def get_connection(database_name=None):
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=database_name or MYSQL_DATABASE,
        autocommit=True,
        charset="utf8mb4",
    )
    return connection


def ensure_database_exists():
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            autocommit=True,
            charset="utf8mb4",
        )
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.close()
        connection.close()
    except Error as exc:
        raise RuntimeError(f"MySQL connection failed: {exc}") from exc


def initialize_database():
    ensure_database_exists()
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        )
        """
    )
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s) ON DUPLICATE KEY UPDATE username = username",
        ("admin", "admin123"),
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            news_title TEXT,
            news_content LONGTEXT,
            selected_model VARCHAR(100),
            prediction VARCHAR(50),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            accuracy DOUBLE,
            precision_value DOUBLE,
            recall_value DOUBLE,
            f1_value DOUBLE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.close()
    connection.close()


def validate_login(username, password):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, username, password FROM users WHERE username = %s AND password = %s LIMIT 1",
        (username, password),
    )
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user is not None


def insert_prediction_record(news_title, news_content, selected_model, prediction):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO prediction_history (news_title, news_content, selected_model, prediction) VALUES (%s, %s, %s, %s)",
        (news_title, news_content, selected_model, prediction),
    )
    cursor.close()
    connection.close()


def fetch_prediction_history(page=1, per_page=10):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    offset = (page - 1) * per_page
    cursor.execute("SELECT COUNT(*) AS total FROM prediction_history")
    total = cursor.fetchone()["total"]
    cursor.execute(
        "SELECT id, news_title, selected_model, prediction, created_at FROM prediction_history ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (per_page, offset),
    )
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return rows, total


def save_model_metrics(metrics):
    connection = get_connection()
    cursor = connection.cursor()
    for metric in metrics:
        cursor.execute(
            "INSERT INTO model_metrics (model_name, accuracy, precision_value, recall_value, f1_value) VALUES (%s, %s, %s, %s, %s)",
            (
                metric["model_name"],
                metric["accuracy"],
                metric["precision"],
                metric["recall"],
                metric["f1_score"],
            ),
        )
    cursor.close()
    connection.close()


def fetch_model_metrics():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        "SELECT model_name, accuracy, precision_value, recall_value, f1_value FROM model_metrics ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    recent = {}
    for row in rows:
        normalized = {
            "model_name": row["model_name"],
            "accuracy": row["accuracy"],
            "precision": row["precision_value"],
            "recall": row["recall_value"],
            "f1_score": row["f1_value"],
        }
        recent[row["model_name"]] = normalized
    return recent
