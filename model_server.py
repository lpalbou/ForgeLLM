#!/usr/bin/env python3
"""
Model server wrapper script.
This script serves as a wrapper to call the actual model server implementation.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the actual server
from forgellm.server.main import main

if __name__ == "__main__":
    main() 