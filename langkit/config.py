import os

# Cache location
DEFAULT_XDG_CACHE_HOME: str = "~/.cache"
XDG_CACHE_HOME: str = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_CACHE_HOME: str = os.path.join(XDG_CACHE_HOME, "langkit")

LANGKIT_CACHE: str = os.path.expanduser(os.getenv("LANGKIT_CACHE", DEFAULT_CACHE_HOME))
