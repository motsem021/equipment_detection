# PostgreSQL DB Writer for Equipment Activity Project

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import List

import psycopg2
from psycopg2 import pool

log = logging.getLogger("db_writer")
logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 10


class DBWriter:
    _INSERT_SQL = """
    INSERT INTO equipment_events (
        event_time,
        frame_id,
        equipment_id,
        equipment_class,
        current_state,
        current_activity,
        motion_source,
        total_tracked_seconds,
        total_active_seconds,
        total_idle_seconds,
        utilization_percent
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s
    )
    """

    def __init__(self, host, port, database, user, password):
        self._buffer: List[tuple] = []
        self._lock = threading.Lock()

        self._pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
        )

        log.info("Connected to PostgreSQL")

    def insert(self, payload: dict):
        """
        payload example:
        {
            "frame_id": 450,
            "equipment_id": "EX-001",
            "equipment_class": "excavator",
            "utilization": {
                "current_state": "ACTIVE",
                "current_activity": "DIGGING",
                "motion_source": "arm_only"
            },
            "time_analytics": {
                "total_tracked_seconds": 15.0,
                "total_active_seconds": 12.5,
                "total_idle_seconds": 2.5,
                "utilization_percent": 83.3
            }
        }
        """

        util = payload["utilization"]
        ta = payload["time_analytics"]

        row = (
            datetime.utcnow(),
            payload["frame_id"],
            payload["equipment_id"],
            payload["equipment_class"],
            util["current_state"],
            util["current_activity"],
            util["motion_source"],
            ta["total_tracked_seconds"],
            ta["total_active_seconds"],
            ta["total_idle_seconds"],
            ta["utilization_percent"],
        )

        with self._lock:
            self._buffer.append(row)

            if len(self._buffer) >= BATCH_SIZE:
                self._flush()

    def _flush(self):
        if not self._buffer:
            return

        conn = self._pool.getconn()

        try:
            with conn.cursor() as cur:
                cur.executemany(self._INSERT_SQL, self._buffer)

            conn.commit()
            log.info(f"Inserted {len(self._buffer)} rows")
            self._buffer.clear()

        except Exception as e:
            conn.rollback()
            log.error(f"Insert failed: {e}")

        finally:
            self._pool.putconn(conn)

    def close(self):
        with self._lock:
            self._flush()

        self._pool.closeall()
        log.info("PostgreSQL connection pool closed")
```

Use it like this inside your project:

```python
# Create once at startup
writer = DBWriter(
    host="localhost",
    port=5432,
    database="equipment_db",
    user="postgres",
    password="YOUR_PASSWORD"
)

# Example payload generated from each equipment object
payload = {
    "frame_id": frame_id,
    "equipment_id": f"EX-{eq.track_id:03d}",
    "equipment_class": eq.cls_name,
    "utilization": {
        "current_state": activity_state,
        "current_activity": activity_name,
        "motion_source": motion_source,
    },
    "time_analytics": {
        "total_tracked_seconds": eq.total_time,
        "total_active_seconds": eq.active_time,
        "total_idle_seconds": eq.idle_time,
        "utilization_percent": round(
            (eq.active_time / max(eq.total_time, 1e-6)) * 100,
            1
        )
    }
}

writer.insert(payload)

# Before program exits
writer.close()
