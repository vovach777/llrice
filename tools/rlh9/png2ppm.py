# PNG -> PPM (RGB) for rlh9_train: PNG2PPM="python3 png2ppm.py"; exit 1 on an unreadable file
import sys
from PIL import Image
try:
    Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2], 'PPM')
except Exception:
    sys.exit(1)
