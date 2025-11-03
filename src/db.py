import csv
import sqlite3

DB_PATH = "data/tickets.db"
CSV_PATH = "data/dataset-tickets-multi-lang-5-2-50-version.csv"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_ticket(subject: str, body: str, team: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tickets (subject, body, team) VALUES (?, ?, ?)",
            (subject, body, team),
        )
        conn.commit()
        return cursor.lastrowid

def get_all_tickets():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets ORDER BY id DESC")
        return cursor.fetchall()

def get_ticket(ticket_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        return cursor.fetchone()

def update_ticket(ticket_id: int, subject: str, body: str, team: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tickets SET subject = ?, body = ?, team = ? WHERE id = ?",
            (subject, body, team, ticket_id),
        )
        conn.commit()
        return cursor.rowcount > 0

def delete_ticket(ticket_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
        conn.commit()
        return cursor.rowcount > 0

def import_data(csv_path):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                body TEXT,
                team TEXT
            )
            """
        )
    with open(
        csv_path,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] == "en":
                cursor.execute(
                    "INSERT INTO tickets (subject, body, team) VALUES (?, ?, ?)",
                    (row["subject"], row["body"], row["queue"]),
                )
        conn.commit()

# import_data(CSV_PATH)