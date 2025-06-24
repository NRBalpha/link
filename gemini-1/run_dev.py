#!/usr/bin/env python3
"""
Development server runner for Link AI Chat
This script provides a clean way to run the server with auto-reload
"""

import os
import sys
import subprocess
import signal
import time

def kill_existing_processes():
    """Kill any existing processes on port 5000"""
    try:
        result = subprocess.run(['lsof', '-ti:5000'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split()
            print(f"Found {len(pids)} process(es) using port 5000, terminating...")
            for pid in pids:
                try:
                    subprocess.run(['kill', '-TERM', pid], capture_output=True)
                except:
                    pass
            time.sleep(1)
            
            # Check if any are still running and force kill
            result = subprocess.run(['lsof', '-ti:5000'], capture_output=True, text=True)
            if result.stdout.strip():
                remaining_pids = result.stdout.strip().split()
                print(f"Force killing {len(remaining_pids)} remaining process(es)...")
                for pid in remaining_pids:
                    try:
                        subprocess.run(['kill', '-9', pid], capture_output=True)
                    except:
                        pass
        print("✅ Port 5000 is ready")
    except Exception as e:
        print(f"Note: Could not check port 5000: {e}")

def run_server():
    """Run the Flask development server"""
    print("🚀 Starting Link AI Chat Development Server...")
    print("📝 Auto-reload enabled - server will restart on code changes")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Set environment variables for Flask
    env = os.environ.copy()
    env['FLASK_APP'] = 'chat.py'
    env['FLASK_ENV'] = 'development'
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'chat.py'], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == '__main__':
    run_server() 