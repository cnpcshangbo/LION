import sys
import os

def main():
    print("Hello from Docker!")
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    print("Directory contents:", os.listdir('.'))

if __name__ == "__main__":
    main() 