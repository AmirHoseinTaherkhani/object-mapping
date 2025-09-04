#!/usr/bin/env python3
"""
UI Launcher Script
Starts the Streamlit point selection interface
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch Streamlit app with proper configuration"""
    ui_script = Path(__file__).parent.parent / "object_detection" / "ui" / "point_selector.py"
    
    cmd = [
        "streamlit", "run", str(ui_script),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--theme.base", "light"
    ]
    
    print("🚀 Starting Point Selection UI...")
    print(f"📍 Access at: http://localhost:8501")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
