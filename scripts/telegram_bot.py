"""
Temporary helper to discover your Telegram user ID.

Long-polls Telegram's getUpdates endpoint and prints the chat/user ID of any
message sent to the bot. Once you have your ID, save it elsewhere and stop
running this script — the rest of the project only needs to *send* messages,
which is a single HTTP call that does not require polling.

Setup:
  1. Talk to @BotFather on Telegram, create a bot, copy the token.
  2. Add `TELEGRAM_BOT_TOKEN=<token>` to your .env file.
  3. Run: python3 scripts/telegram_bot.py
  4. In Telegram, send any message to your bot. The script prints your user ID.
"""

import logging
import os
import sys

import requests
from dotenv import load_dotenv

API_BASE = "https://api.telegram.org/bot{token}/{method}"
LONG_POLL_TIMEOUT = 30


def call(token: str, method: str, **params):
    resp = requests.get(
        API_BASE.format(token=token, method=method),
        params=params,
        timeout=LONG_POLL_TIMEOUT + 5,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API error: {payload}")
    return payload["result"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN is not set in the environment / .env")

    me = call(token, "getMe")
    logging.info("Connected as @%s (bot id=%s)", me.get("username"), me.get("id"))
    logging.info("Send any message to the bot in Telegram to see your user ID.")

    offset = None
    while True:
        params = {"timeout": LONG_POLL_TIMEOUT}
        if offset is not None:
            params["offset"] = offset
        updates = call(token, "getUpdates", **params)
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message") or update.get("edited_message") or {}
            sender = msg.get("from", {})
            chat = msg.get("chat", {})
            logging.info(
                "from.id=%s  chat.id=%s  username=%s  text=%r",
                sender.get("id"),
                chat.get("id"),
                sender.get("username"),
                msg.get("text"),
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
