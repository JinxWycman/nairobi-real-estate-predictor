import os
import sys
import subprocess

def main():
    # Get port from environment or use default
    port = os.environ.get('PORT', '8501')
    
    print(f"🚀 Starting Streamlit app on port: {port}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if app file exists
    app_path = "app/streamlit_app.py"
    if not os.path.exists(app_path):
        print(f"❌ ERROR: {app_path} not found!")
        print("Files in current directory:")
        for root, dirs, files in os.walk("."):
            for file in files:
                print(f"  {os.path.join(root, file)}")
        sys.exit(1)
    
    # Run Streamlit
    cmd = [
        "streamlit", "run",
        app_path,
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    print(f"📝 Command: {' '.join(cmd)}")
    
    # Execute
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
