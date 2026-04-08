from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    sam_predictor: Any = None
    sessoes: dict[str, dict] = field(default_factory=dict)


state = AppState()

