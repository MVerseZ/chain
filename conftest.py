# conftest.py
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы pytest видел agent.py
sys.path.insert(0, str(Path(__file__).resolve().parent))