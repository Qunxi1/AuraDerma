from __future__ import annotations

from textwrap import dedent

from core import JsonParser, get_logger
from prompts import SYSTEM_PROMPT, WEATHER_EXTRACT_PROMPT

log = get_logger("auraderma.weather")


class WeatherService:
    """天气查询服务。

    提取用户城市并获取气候数据，用于气候适应性护肤建议。
    """

    def __init__(self, llm, skill_manager):
        self._llm = llm
        self._skill_manager = skill_manager

    def fetch_weather(self, question: str, profile_lines: list[str]) -> str:
        """提取城市 + 查询天气。

        Args:
            question: 用户问题
            profile_lines: 用户画像摘要

        Returns:
            格式化后的天气字符串，或空字符串（无法确定城市时）
        """
        log.info("尝试获取天气数据")

        city = self._extract_city(question, profile_lines)
        if not city:
            log.info("无法确定用户城市，跳过天气查询")
            return ""

        log.info("获取城市天气: city=%s", city)
        try:
            weather = self._skill_manager.weather.fetch(city)
            result = (
                f"用户所在城市：{weather.get('city', city)}"
                f"{' (' + weather['region'] + ')' if weather.get('region') else ''}\n"
                f"当前气温：{weather['temperature']}°C"
                f" (体感 {weather['feels_like']}°C)\n"
                f"当前湿度：{weather['humidity']}%\n"
                f"天气状况：{weather['condition']}"
            )
            log.info("天气查询成功: city=%s temp=%s", city, weather.get('temperature'))
            return result
        except Exception as e:
            log.warning("天气查询失败: city=%s error=%s", city, e)
            return f"天气查询失败: {e}"

    def _extract_city(self, question: str, profile_lines: list[str]) -> str | None:
        """从问题或记忆画像中提取城市。"""
        extract_prompt = dedent(f"""
            用户问题:
            {question}

            用户画像记忆:
            {chr(10).join(profile_lines) if profile_lines else '无'}

            请判断用户所在城市，返回 JSON。
        """).strip()
        raw = self._llm.chat(
            SYSTEM_PROMPT,
            f"{WEATHER_EXTRACT_PROMPT}\n\n{extract_prompt}",
            temperature=0.0,
        )
        obj = JsonParser.safe_parse_obj(raw, context="weather_extract")
        return obj.get("city") or None
