"""
SQLite Database Module for PhishGuard AI
Provides persistent storage for scan history.
Database file: data/phishguard.db (auto-created)
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "phishguard.db"


def _get_connection():
    """Get a SQLite connection with row factory."""
    os.makedirs(str(DB_PATH.parent), exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database and create tables if they don't exist.
    Safe to call multiple times.
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT NOT NULL,
                input_text TEXT NOT NULL,
                score REAL DEFAULT 0,
                is_phishing INTEGER DEFAULT 0,
                risk_level TEXT DEFAULT 'LOW',
                timestamp TEXT NOT NULL,
                full_result TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        print(f"[Database] Initialized at {DB_PATH}")
    finally:
        conn.close()


def save_scan(scan_type, input_text, score, is_phishing, risk_level, full_result=None):
    """
    Save a scan result to the database.

    Args:
        scan_type (str): 'sms', 'url', or 'fullscan'
        input_text (str): The message or URL that was scanned
        score (float): Threat score (0-1 or 0-100)
        is_phishing (bool): Whether phishing was detected
        risk_level (str): 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
        full_result (dict): Full JSON result from the analysis

    Returns:
        int: The ID of the inserted record
    """
    conn = _get_connection()
    try:
        # Normalize score to 0-100 range
        if score is not None and score <= 1:
            score = round(score * 100, 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_json = json.dumps(full_result) if full_result else "{}"

        cursor = conn.execute(
            """INSERT INTO scan_history
               (scan_type, input_text, score, is_phishing, risk_level, timestamp, full_result)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (scan_type, input_text[:500], score, int(is_phishing), risk_level, timestamp, result_json)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_history(limit=50):
    """
    Retrieve scan history from the database, newest first.

    Args:
        limit (int): Maximum number of records to return

    Returns:
        list[dict]: List of scan history records
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM scan_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

        history = []
        for row in rows:
            entry = {
                "id": row["id"],
                "scanType": row["scan_type"],
                "input": row["input_text"] if len(row["input_text"]) <= 50
                         else row["input_text"][:50] + "…",
                "score": row["score"],
                "isPhishing": bool(row["is_phishing"]),
                "riskLevel": row["risk_level"],
                "time": row["timestamp"],
            }
            history.append(entry)

        return history
    finally:
        conn.close()


def clear_history():
    """Delete all scan history records."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM scan_history")
        conn.commit()
        print("[Database] History cleared")
    finally:
        conn.close()


# Quick self-test
if __name__ == "__main__":
    init_db()
    rid = save_scan("sms", "Test phishing message", 0.87, True, "HIGH")
    print(f"Saved scan with id={rid}")
    history = get_history(5)
    print(f"History ({len(history)} records):")
    for h in history:
        print(f"  {h}")
    clear_history()
    print("Self-test complete!")
