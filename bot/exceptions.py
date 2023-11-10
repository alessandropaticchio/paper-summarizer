class ServiceNotAvailable(Exception):
    def __init__(self, service_name: str, status_code: int, error_string: str):
        super().__init__()
        self.service_name = service_name
        self.status_code = status_code
        self.error_string = error_string

    def __str__(self) -> str:
        return f"External service {self.service_name} returned {self.status_code} with the following error: {self.error_string}"
