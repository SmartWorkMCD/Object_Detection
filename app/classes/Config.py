from dataclasses import dataclass
from .IPAddress import IPAddress


@dataclass
class Config:
    BROKER_IP: IPAddress = "127.0.0.1"
    BROKER_PORT: int = 1883
    BROKER_USER: str = "user"
    BROKER_PASSWORD: str = "password"
    BROKER_TOPIC: str = "objdet/results"

    VERBOSE: bool = False
