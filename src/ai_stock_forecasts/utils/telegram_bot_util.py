import logging
import os

import requests
from dotenv import load_dotenv

API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramBotUtil:
    def __init__(self):
        load_dotenv()
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_id = os.environ["TELEGRAM_CHAT_ID"]

    def send_message(self, message: str) -> None:
        resp = requests.post(
            API_URL.format(token=self.token),
            data={"chat_id": self.chat_id, "text": message},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")
        logging.info("sent telegram message to chat_id=%s", self.chat_id)
