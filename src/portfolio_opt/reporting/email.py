from __future__ import annotations

from pathlib import Path

import yagmail

from ..config import load_settings


def send_email(subject: str, body: str, attachments: list[str | Path] | None = None) -> None:
    settings = load_settings()
    user = settings.get("EMAIL_USER")
    app_pw = settings.get("EMAIL_APP_PASSWORD")
    if not user or not app_pw:
        print("[email] Skipping: EMAIL_USER or EMAIL_APP_PASSWORD not set.")
        return
    yag = yagmail.SMTP(user=user, password=app_pw)
    yag.send(to=user, subject=subject, contents=body, attachments=attachments or [])
    print("[email] Sent.")
