from pathlib import Path

CODE_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_CONFIG = CODE_DIR / 'inputs' / 'variables.toml'
DEFAULT_LIMITS = CODE_DIR / 'inputs' / 'limits.toml'
DEFAULT_IMG_DIR = CODE_DIR / 'outputs'
