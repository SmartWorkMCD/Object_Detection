from config import MQTT_CONFIG
import time
import paho.mqtt.client as mqtt


def log_message(message: str, *args, level: str = "INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not MQTT_CONFIG.VERBOSE:
        if level != "INFO":
            print(f"{timestamp} [{level}] {message}", *args)
        return

    print(f"{timestamp} [{level}] {message}", *args)


def init_broker() -> mqtt.Client:
    """Initialize the MQTT broker client with the provided configuration."""
    client = mqtt.Client()
    client.username_pw_set(MQTT_CONFIG.BROKER_USER, MQTT_CONFIG.BROKER_PASSWORD)
    return client


def connect_broker(client: mqtt.Client, retries: int = 10, delay: int = 2) -> None:

    #"""Connect to the MQTT broker."""
    #client.connect(MQTT_CONFIG.BROKER_IP, MQTT_CONFIG.BROKER_PORT)
    #log_message(f"Connected to broker at {MQTT_CONFIG.BROKER_IP}:{MQTT_CONFIG.BROKER_PORT}")

    for attempt in range(retries):
        try:
            client.connect(MQTT_CONFIG.BROKER_IP, MQTT_CONFIG.BROKER_PORT)
            log_message(f"Connected to broker at {MQTT_CONFIG.BROKER_IP}:{MQTT_CONFIG.BROKER_PORT}")
            return
        except Exception as e:
            log_message(f"Connection attempt {attempt + 1} failed: {e}", level="ERROR")
            time.sleep(delay)
    raise ConnectionError("Failed to connect to MQTT broker after retries.")

