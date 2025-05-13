class IPAddress(str):
    def __new__(cls, value: str):
        if not cls.is_valid(value):
            raise ValueError(f"Invalid IP address: {value}")
        return str.__new__(cls, value)

    @staticmethod
    def is_valid(ip: str) -> bool:
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        for part in parts:
            if not part.isdigit() or not (0 <= int(part) <= 255):
                return False
        return True
