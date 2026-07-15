"""Shared pytest setup: put src/ on the import path (the project's scripts
import each other directly, e.g. `from train_multitask import ...`, matching
the convention of always running from inside src/) and quiet Streamlit's
noisy "missing ScriptRunContext" logging that fires when app.py is imported
outside `streamlit run` (harmless -- see README's Running tests section).
"""
import logging
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

logging.getLogger("streamlit").setLevel(logging.ERROR)
