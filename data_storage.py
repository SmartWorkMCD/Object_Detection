import sqlite3
from datetime import datetime

DB_NAME = "aggregator.db"

def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            assembly_time REAL NOT NULL,
            defect_count INTEGER NOT NULL,
            defect_type TEXT,
            success BOOLEAN NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_log(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (station_id, timestamp, assembly_time, defect_count, defect_type, success)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data['station_id'],
        data['timestamp'],
        data['assembly_time'],
        data['defect_count'],
        data['defect_type'],
        data['success']
    ))
    conn.commit()
    conn.close()
