"""Local OCR adapters. Images travel through stdin, never a cloud service."""
from io import BytesIO
from pathlib import Path
from functools import lru_cache
import json
import platform
import shutil
import subprocess
import numpy as np


def apple_ocr(image, fast=False):
    binary = Path(__file__).resolve().parents[1] / "tmp/vision-ocr"
    if platform.system() != "Darwin" or not binary.exists():
        return dict(available=False, tokens=[], message="Local OCR is not installed. On macOS run the setup command in SETUP.md.")
    buf = BytesIO()
    # Enlarge small text while retaining normalized coordinates.
    im = image.convert("RGB")
    scale = min(3, 1800 / max(im.size))
    if scale > 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    im.save(buf, format="PNG")
    try:
        command = [str(binary)] + (["--fast"] if fast else [])
        result = subprocess.run(command, input=buf.getvalue(), capture_output=True, timeout=30, check=True)
        tokens=json.loads(result.stdout)
        for token in tokens:
            token["engine"]="Apple Vision"
        return dict(available=True, tokens=tokens, message="Text read locally using Apple Vision.")
    except (subprocess.SubprocessError, OSError, ValueError):
        return dict(available=False, tokens=[], message="Local text recognition could not read this image.")


@lru_cache(maxsize=1)
def rapid_engine():
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from rapidocr_onnxruntime import RapidOCR
    # Weights ship inside the installed wheel; inference makes no download calls.
    return RapidOCR(intra_op_num_threads=2,inter_op_num_threads=2)


def rapid_ocr(image):
    try:
        engine=rapid_engine()
    except ImportError:
        return dict(available=False,tokens=[],message="Install requirements-ocr.txt for portable local OCR, or set up Apple Vision on macOS.")
    except (RuntimeError, ValueError, OSError):
        return dict(available=False,tokens=[],message="The installed local OCR engine could not be initialized.")
    im=image.convert("RGB")
    factor=min(3,1800/max(im.size))
    im=im.resize((round(im.width*factor),round(im.height*factor)))
    try:
        results,_=engine(np.asarray(im)[:,:,::-1].copy())
    except (RuntimeError, ValueError, OSError):
        return dict(available=False,tokens=[],message="The local OCR engine could not read this image.")
    tokens=[]
    for corners,text,score in results or []:
        xy=np.asarray(corners,dtype=float)
        tokens.append(dict(text=text,confidence=float(score),engine="RapidOCR",box=[float(xy[:,0].min()/im.width),float(xy[:,1].min()/im.height),float(xy[:,0].max()/im.width),float(xy[:,1].max()/im.height)]))
    return dict(available=True,tokens=tokens,message="Text read locally using RapidOCR.")


def recognize(image, fast=False, engine="auto"):
    if engine=="rapid":
        return rapid_ocr(image)
    native=apple_ocr(image,fast=fast)
    return native if native["available"] or engine=="apple" else rapid_ocr(image)


def setup():
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Darwin" or not shutil.which("swiftc"):
        raise RuntimeError("Apple Vision OCR requires macOS and the Xcode command-line tools.")
    cache = root / "tmp/ocr-cache"
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(["swiftc", "-module-cache-path", str(cache), str(root / "src/ocr.swift"), "-o", str(root / "tmp/vision-ocr")], check=True)


if __name__ == "__main__":
    setup()
