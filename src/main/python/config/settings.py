"""配置层——薄包装，实际实现在 core/settings.py。"""
from src.main.python.core.settings import Settings, reload_settings, settings

__all__ = ["Settings", "reload_settings", "settings"]
