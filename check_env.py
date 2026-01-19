import sys
import importlib.util

packages = ['gymnasium', 'retro', 'torch', 'torchvision', 'cv2', 'numpy', 'matplotlib']
print(f"Python executable: {sys.executable}")
print("-" * 20)
for package in packages:
    spec = importlib.util.find_spec(package)
    found = "✅ Found" if spec else "❌ Missing"
    if spec and package == 'retro':
        print(f"{package}: {found} ({spec.origin})")
    else:
        print(f"{package}: {found}")
