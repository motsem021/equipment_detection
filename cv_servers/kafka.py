


# Kafka Publisher for Equipment Activity Events

from __future__ import annotations

import json
import logging

from confluent_kafka import Producer

log = logging.getLogger("kafka_pub")
logging.basicConfig(level=logging.INFO)


class KafkaPublisher:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic

        self._producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "equipment-cv-service",
            "acks": "1",
            "compression.type": "lz4",
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.ms": 50,
        })

        log.info(f"Connected to Kafka: {bootstrap_servers} -> {topic}")

    def publish_equipment_event(self, eq, frame_id, activity, motion_source="unknown"):
        payload = {
            "frame_id": frame_id,
            "equipment_id": f"EX-{eq.track_id:03d}",
            "track_id": eq.track_id,
            "equipment_class": eq.cls_name,
            "timestamp": frame_id,
            "utilization": {
                "current_state": "ACTIVE" if activity not in ["WAITING", "IDLE", "INACTIVE"] else "IDLE",
                "current_activity": activity,
                "motion_source": motion_source,
            },
            "time_analytics": {
                "total_tracked_seconds": round(eq.total_time, 2),
                "total_active_seconds": round(eq.active_time, 2),
                "total_idle_seconds": round(eq.idle_time, 2),
                "utilization_percent": round(
                    (eq.active_time / max(eq.total_time, 1e-6)) * 100,
                    1,
                ),
            },
        }

        try:
            self._producer.produce(
                topic=self.topic,
                key=payload["equipment_id"].encode("utf-8"),
                value=json.dumps(payload).encode("utf-8"),
                callback=self._delivery_callback,
            )

            self._producer.poll(0)

        except Exception as e:
            log.error(f"Kafka publish failed: {e}")

    def close(self):
        self._producer.flush(5)
        log.info("Kafka producer closed")

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            log.error(f"Delivery failed: {err}")
        else:
            log.info(
                f"Delivered event for {msg.key().decode()} "
                f"to partition {msg.partition()}"
            )

publisher = KafkaPublisher(
    bootstrap_servers="localhost:9092",
    topic="equipment-events"
)



