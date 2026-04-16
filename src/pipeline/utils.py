import time
import uuid
import requests


class GigaChatTokenProvider:
    def __init__(
        self,
        auth_token: str,
        auth_url: str,
        scope: str,
        verify_ssl: bool = False
    ):
        self.auth_token = auth_token
        self.auth_url = auth_url
        self.scope = scope
        self.verify_ssl = verify_ssl

        self.access_token = None
        self.expires_at = 0  # unix timestamp ms


    def _request_new_token(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_token}",
        }

        payload = {"scope": self.scope}

        resp = requests.post(
            self.auth_url,
            headers=headers,
            data=payload,
            timeout=30,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()

        data = resp.json()
        self.access_token = data["access_token"]
        self.expires_at = data["expires_at"]


    def get_token(self) -> str:
        now = int(time.time() * 1000)
        if not self.access_token or now >= self.expires_at - 3000:
            self._request_new_token()
        return self.access_token