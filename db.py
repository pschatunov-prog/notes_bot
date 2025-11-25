import sqlite3
import datetime
from typing import List, Dict, Optional

DB_NAME = "notes.db"

class Database:
    def __init__(self, db_name=DB_NAME):
        self.db_name = db_name
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def init_db(self):
        """Initialize the database with the notes table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_note(self, user_id: int, content: str, summary: str, tags: str) -> int:
        """Add a new note to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO notes (user_id, content, summary, tags)
            VALUES (?, ?, ?, ?)
        ''', (user_id, content, summary, tags))
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return note_id

    def get_notes(self, user_id: int) -> List[Dict]:
        """Retrieve all notes for a specific user."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM notes WHERE user_id = ? ORDER BY created_at DESC
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_notes_text(self, user_id: int) -> str:
        """Get all notes as a single string for analysis."""
        notes = self.get_notes(user_id)
        if not notes:
            return ""
        return "\n".join([f"ID: {n['id']} | Date: {n['created_at']} | Content: {n['content']} | Tags: {n['tags']}" for n in notes])
