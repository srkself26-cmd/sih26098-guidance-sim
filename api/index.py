import os
import sys

# Add parent directory to sys.path to enable imports of root modules on Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app import app
