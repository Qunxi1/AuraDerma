from __future__ import annotations

import sys


class ProgressReporter:
    """打印简洁的阶段性提示，让用户知道当前在处理什么。"""

    def step(self, icon: str, msg: str) -> None:
        print(f"  {icon} {msg}", flush=True)

    def thinking(self, label: str = "") -> None:
        suffix = f" {label}" if label else ""
        self.step("🤔", f"推理中{suffix}...")

    def intent(self) -> None:
        self.step("🎯", "分析意图...")

    def workflow(self) -> None:
        self.step("📋", "规划流程...")

    def regimen(self) -> None:
        self.step("📋", "规划护肤体系...")

    def product_search(self, query: str = "") -> None:
        suffix = f"：{query}" if query else "..."
        self.step("📦", f"搜索产品{suffix}")

    def memory(self) -> None:
        self.step("📖", "读取记忆...")

    def weather(self, city: str = "") -> None:
        suffix = f"（{city}）" if city else ""
        self.step("🌤", f"查询天气{suffix}")

    def web_search(self, query: str = "") -> None:
        suffix = f"：{query}" if query else "..."
        self.step("🔍", f"搜索网页{suffix}")

    def skill(self, name: str) -> None:
        self.step("🔧", f"调用技能：{name}")

    def answer(self) -> None:
        self.step("✨", "生成回答...")


# 全局空报告器，不打印任何内容
class NullReporter(ProgressReporter):
    def step(self, icon: str, msg: str) -> None:
        pass
