#!/usr/bin/env python3
"""
Launch the integrated Video-to-Map webapp
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the webapp directory
    webapp_dir = Path(__file__).parent.parent / "webapp"
    main_file = webapp_dir / "main.py"
    
    if not main_file.exists():
        print(f"Error: {main_file} not found")
        return 1
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(main_file),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    print("Starting Video-to-Map webapp...")
    print(f"Access at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, cwd=webapp_dir.parent.parent)
    except KeyboardInterrupt:
        print("\nWebapp stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())
