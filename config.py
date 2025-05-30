# config.py

INPUT_SCHEMA = {
    "station_id": str,
    "timestamp": str,  # formato ISO 8601: '2025-05-30T15:00:00'
    "assembly_time": float,
    "defect_count": int,
    "defect_type": str,
    "success": bool
}