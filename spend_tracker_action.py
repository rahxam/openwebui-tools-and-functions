"""
title: Show credit balance
author: Maximilian Hartig
version: 0.0.1
required_open_webui_version: 0.5.0
icon_url: data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgoKPHN2ZwogICB3aWR0aD0iMjRtbSIKICAgaGVpZ2h0PSIyNG1tIgogICB2aWV3Qm94PSIwIDAgMjQgMjQiCiAgIHZlcnNpb249IjEuMSIKICAgaWQ9InN2ZzUiCiAgIHhtbDpzcGFjZT0icHJlc2VydmUiCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnMKICAgICBpZD0iZGVmczIiIC8+PHRleHQKICAgICB4bWw6c3BhY2U9InByZXNlcnZlIgogICAgIHN0eWxlPSJmb250LXNpemU6My4yMTgzN3B4O2ZvbnQtZmFtaWx5OlZpcmdpbDstaW5rc2NhcGUtZm9udC1zcGVjaWZpY2F0aW9uOlZpcmdpbDtmaWxsOiM2NzY3Njc7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlLXdpZHRoOjAuNjA4NTQxO3N0cm9rZS1saW5lY2FwOnJvdW5kO3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtZGFzaGFycmF5Om5vbmUiCiAgICAgeD0iMTYuNTE3MTkxIgogICAgIHk9Ii0xLjk1NjE0MTUiCiAgICAgaWQ9InRleHQ0NTciCiAgICAgdHJhbnNmb3JtPSJzY2FsZSgwLjk5MzQyNjk4LDEuMDA2NjE2NSkiPjx0c3BhbgogICAgICAgaWQ9InRzcGFuNDU1IgogICAgICAgc3R5bGU9ImZpbGw6IzY3Njc2NztmaWxsLW9wYWNpdHk6MTtzdHJva2Utd2lkdGg6MC42MDg1NDE7c3Ryb2tlLWRhc2hhcnJheTpub25lIgogICAgICAgeD0iMTYuNTE3MTkxIgogICAgICAgeT0iLTEuOTU2MTQxNSIgLz48L3RleHQ+PGcKICAgICBpZD0iZzE4NjkxIgogICAgIHRyYW5zZm9ybT0ibWF0cml4KDAuODI4MTEzOTgsMCwwLDAuODMxMDI1NzMsMi4zNDAyODg3LDIuMzcwOTM0MSkiCiAgICAgc3R5bGU9InN0cm9rZS13aWR0aDoxLjIwNTQ1Ij48dGV4dAogICAgICAgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIKICAgICAgIHN0eWxlPSJmb250LXNpemU6MTQuMjQ1cHg7Zm9udC1mYW1pbHk6VmlyZ2lsOy1pbmtzY2FwZS1mb250LXNwZWNpZmljYXRpb246VmlyZ2lsO2ZpbGw6IzY3Njc2NztmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MC43MzM1NjM7c3Ryb2tlLWxpbmVjYXA6cm91bmQ7c3Ryb2tlLWxpbmVqb2luOnJvdW5kO3N0cm9rZS1kYXNoYXJyYXk6bm9uZTtzdHJva2Utb3BhY2l0eToxIgogICAgICAgeD0iMTIuODE3OTE5IgogICAgICAgeT0iMTkuNTEwNDg5IgogICAgICAgaWQ9InRleHQxMTcxIgogICAgICAgdHJhbnNmb3JtPSJzY2FsZSgwLjk5MzQyNjk5LDEuMDA2NjE2NSkiPjx0c3BhbgogICAgICAgICBpZD0idHNwYW4xMTY5IgogICAgICAgICBzdHlsZT0iZm9udC1zdHlsZTpub3JtYWw7Zm9udC12YXJpYW50Om5vcm1hbDtmb250LXdlaWdodDpib2xkO2ZvbnQtc3RyZXRjaDpub3JtYWw7Zm9udC1mYW1pbHk6J0RhcnRtb3V0aCBSdXppY2thJzstaW5rc2NhcGUtZm9udC1zcGVjaWZpY2F0aW9uOidEYXJ0bW91dGggUnV6aWNrYSBCb2xkJztmaWxsOiM2NzY3Njc7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjAuNzMzNTYzO3N0cm9rZS1kYXNoYXJyYXk6bm9uZTtzdHJva2Utb3BhY2l0eToxIgogICAgICAgICB4PSIxMi44MTc5MTkiCiAgICAgICAgIHk9IjE5LjUxMDQ4OSI+JDwvdHNwYW4+PC90ZXh0PjxyZWN0CiAgICAgICBzdHlsZT0iZmlsbDpub25lO2ZpbGwtb3BhY2l0eToxO3N0cm9rZTojNjc2NzY3O3N0cm9rZS13aWR0aDoxLjkxMzY0O3N0cm9rZS1saW5lY2FwOmJ1dHQ7c3Ryb2tlLWxpbmVqb2luOnJvdW5kO3N0cm9rZS1kYXNoYXJyYXk6bm9uZTtzdHJva2Utb3BhY2l0eToxIgogICAgICAgaWQ9InJlY3QyMDYxIgogICAgICAgd2lkdGg9IjE3LjU4OTg1MSIKICAgICAgIGhlaWdodD0iMjIuMTE0MjA2IgogICAgICAgeD0iMy4yMDUwNzQxIgogICAgICAgeT0iMC45NDI4OTYzNyIgLz48cGF0aAogICAgICAgc3R5bGU9ImZpbGw6bm9uZTtmaWxsLW9wYWNpdHk6MTtzdHJva2U6IzY3Njc2NztzdHJva2Utd2lkdGg6MS45MTM2NDtzdHJva2UtbGluZWNhcDpidXR0O3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLW9wYWNpdHk6MSIKICAgICAgIGQ9Ik0gNi40NjI0NTM1LDMuODU4NDUxMSBIIDE3LjUzNzU0NiIKICAgICAgIGlkPSJwYXRoMjEyNiIgLz48cGF0aAogICAgICAgc3R5bGU9ImZpbGw6bm9uZTtmaWxsLW9wYWNpdHk6MTtzdHJva2U6IzY3Njc2NztzdHJva2Utd2lkdGg6MS45MTM2NDtzdHJva2UtbGluZWNhcDpidXR0O3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLW9wYWNpdHk6MSIKICAgICAgIGQ9Ik0gNi40NjI0NTM1LDcuMTMyMTU4OSBIIDE3LjUzNzU0NiIKICAgICAgIGlkPSJwYXRoMjEyNi01IiAvPjwvZz48L3N2Zz4K
"""

from typing import Optional
import requests
import logging
from datetime import datetime
import time
from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        LITELLM_HOST: str = Field(
            default="https://ai-litellm.niceplant-2c8081d6.germanywestcentral.azurecontainerapps.io",
            description="Die Basis-URL für die LiteLLM API.",
        )
        LITELLM_API_KEY: str = Field(
            default="",
            description="Der API-Schlüssel zur Authentifizierung bei der LiteLLM API.",
            # Dies stellt sicher, dass das Feld im UI als Passwortfeld behandelt wird.
            extra={"type": "password"},
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.is_visible = False
        self.logger = logging.getLogger(__name__)

    def get_daily_cost(self, user_email: str) -> str:
        """
        Ruft die heutigen Gesamtausgaben des Benutzers für die API-Nutzung vom LiteLLM-Proxy ab.
        Diese Funktion wird verwendet, wenn der Benutzer nach seinen heutigen Kosten oder Ausgaben fragt.
        """
        if not user_email:
            return "Fehler: Ihre E-Mail-Adresse konnte nicht ermittelt werden, um die Kosten abzurufen."

        # Überprüfen, ob der API-Schlüssel konfiguriert ist
        if not self.valves.LITELLM_API_KEY:
            return "Fehler: Der LiteLLM API Key ist in den Einstellungen der Funktion nicht konfiguriert."

        today_date = datetime.now().strftime("%Y-%m-%d")

        # URL und Parameter für die Anfrage zusammenstellen
        url = f"{self.valves.LITELLM_HOST}/tag/daily/activity"
        params = {
            "tags": f"x-openwebui-user-email: {user_email}",
            "start_date": today_date,
            "end_date": today_date,
            "page": 1,
            "page_size": 10,
        }
        headers = {"Authorization": f"Bearer {self.valves.LITELLM_API_KEY}"}

        try:
            self.logger.debug("Fetching spend data for user %s", user_email)
            # Senden der GET-Anfrage an die API
            response = requests.get(url, headers=headers, params=params, timeout=15)
            # Löst eine Ausnahme aus, wenn der HTTP-Statuscode ein Fehler ist (4xx oder 5xx)
            response.raise_for_status()

            # Extrahieren der Daten aus der JSON-Antwort
            data = response.json()
            total_spend = data.get("metadata", {}).get("total_spend", 0.0)

            # Formatieren und Zurückgeben der Antwort für den Benutzer
            return f"Deine heutigen Kosten betragen: ${total_spend:.4f}."
            self.logger.debug("Fetched spend data for user %s", user_email)

        except requests.exceptions.HTTPError as e:
            return f"Fehler bei der API-Anfrage: {e.response.status_code} {e.response.reason}. Bitte überprüfe den Host und den API-Schlüssel."
        except requests.exceptions.RequestException as e:
            return f"Fehler beim Abrufen der Kostendaten: {e}. Bitte überprüfe die Netzwerkverbindung und den Host."
        except Exception as e:
            return f"Ein unerwarteter Fehler ist aufgetreten: {e}"

    async def action(
        self,
        body: dict,
        __user__=None,
        __metadata__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        import traceback

        await __event_emitter__(
            {
                "type": "status",
                "action": "credit_balance",
                "data": {"description": "Getting spend data...", "done": False},
            }
        )
        time.sleep(0.5)

        stats = "Fehler"

        try:
            if __user__:
                if "email" in __user__:
                    user_email = __user__["email"]
                else:
                    self.logger.debug("**ERROR: User email not found!")
                try:
                    stats = self.get_daily_cost(user_email)
                except Exception as _:
                    self.logger.debug("**ERROR: Unable to update user cost file!")
            else:
                self.logger.debug("**ERROR: User not found!")

            await __event_emitter__(
                {
                    "type": "status",
                    "action": "credit_balance",
                    "data": {"description": stats, "done": True},
                }
            )
            self.is_visible = True
            self.logger.debug("credit_balance: %s %s", __user__, stats)
            return body
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Error in credit_balance action: {e}\n{tb}")
            await __event_emitter__(
                {
                    "type": "status",
                    "action": "credit_balance",
                    "data": {"description": f"Error: {str(e)}", "done": True},
                }
            )
            return None
