"""
Gestura - Root Entrypoint
python main.py
"""
import os
import sys

# Add 'Asphalt 8' directory to sys.path
asphalt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Asphalt 8")
if asphalt_dir not in sys.path:
    sys.path.insert(0, asphalt_dir)

import main as asphalt_main

if __name__ == "__main__":
    asphalt_main.main()
