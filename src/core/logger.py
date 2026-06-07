from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = (
    "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _default_log_dir() -> Path:
    data_dir = os.getenv("AURADERMA_DATA_DIR", "./data")
    return Path(data_dir).resolve() / "logs"


class AuraDermaLogger:
    """结构化日志管理器，支持控制台 + 文件双输出。

    日志分级：
    - DEBUG:   开发和诊断信息（包括 LLM 原始返回、JSON 解析失败详情）
    - INFO:    正常业务流程信息
    - WARNING: 非预期的但可恢复的情况（如 JSON 修复成功、降级处理）
    - ERROR:   不可恢复的错误（如 LLM 调用彻底失败）
    """

    def __init__(
        self,
        name: str = "auraderma",
        level: int = logging.DEBUG if os.getenv("AURADERMA_DEBUG") else logging.INFO,
        log_dir: str | Path | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        self._logger.propagate = False

        # 控制台 handler（ansi 彩色友好）
        console = logging.StreamHandler(stream or sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        self._logger.addHandler(console)

        # 文件 handler（轮转，最多 5 * 10MB）
        log_dir = Path(log_dir) if log_dir else _default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT_VERBOSE, _DATE_FORMAT))
        self._logger.addHandler(file_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)


# 全局默认日志实例
_default_logger: AuraDermaLogger | None = None


def get_logger(name: str = "auraderma") -> AuraDermaLogger:
    """获取或创建全局 AuraDermaLogger 实例。"""
    global _default_logger
    if _default_logger is None or _default_logger.logger.name != name:
        _default_logger = AuraDermaLogger(name=name)
    return _default_logger
