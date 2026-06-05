from __future__ import annotations

import json
import urllib.parse
import urllib.request


class WeatherSkill:
    """Fetch real-time temperature and humidity for a given city via wttr.in."""

    _BASE_URL = "https://wttr.in"

    def fetch(self, city: str) -> dict:
        """Fetch weather data for a city.

        Returns dict with:
          - city: resolved city name
          - temperature: int °C
          - humidity: int %
          - condition: str description
          - raw: str full wttr.in text (for LLM context if needed)
        """
        raw_json = self._fetch_json(city)
        return self._parse(raw_json, city)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _fetch_json(self, city: str) -> dict:
        encoded = urllib.parse.quote(city)
        url = f"{self._BASE_URL}/{encoded}?format=j1"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"天气查询失败 (HTTP {e.code}): {city}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"天气服务连接失败: {e}") from e

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(data: dict, original_city: str) -> dict:
        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]

        # Resolve display city name
        aname = ""
        if area.get("areaName"):
            aname = area["areaName"][0].get("value", "")
        region = ""
        if area.get("region"):
            region = area["region"][0].get("value", "")

        return {
            "city": aname or original_city,
            "region": region,
            "temperature": int(current.get("temp_C", 0)),
            "humidity": int(current.get("humidity", 0)),
            "condition": current.get("weatherDesc", [{}])[0].get("value", ""),
            "feels_like": int(current.get("FeelsLikeC", 0)),
            "wind_speed": current.get("windspeedKmph", ""),
        }
