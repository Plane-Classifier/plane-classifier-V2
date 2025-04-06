import time
import subprocess

def print_gpu_status():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("\n----- GPU STATUS -----")
        print(output)
    except Exception as e:
        print("Could not retrieve GPU usage:", e)

if __name__ == "__main__":
    print("Monitoring GPU every 5 seconds. Press Ctrl+C to stop.")
    try:
        while True:
            print_gpu_status()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped GPU monitor.")
