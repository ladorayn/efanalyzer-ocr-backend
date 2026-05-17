import os
import gc
import logging
import tempfile
import uvicorn
import easyocr
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EF Analyzer OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize once on startup
# EasyOCR uses ~200MB RAM — fits Railway free tier
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    verbose=False,
    quantize=True,
    model_storage_directory='/tmp/easyocr_models',
)

logger.info("EasyOCR initialized successfully")


def preprocess_image(input_path: str) -> str:
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        w, h = img.size

        # Cap maximum size — prevents OOM on large screenshots
        max_width = 1080
        if w > max_width:
            scale = max_width / w
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )
            logger.info(f"Downscaled from {w}x{h} to {img.size}")
        elif w < 800:
            scale = 800 / w
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )
            logger.info(f"Upscaled from {w}x{h} to {img.size}")

        # Mild contrast only — no sharpness (saves processing)
        img = ImageEnhance.Contrast(img).enhance(1.2)

        preprocessed_path = input_path + '_processed.png'
        img.save(preprocessed_path, 'PNG', optimize=True)

    gc.collect()
    return preprocessed_path
def run_ocr(image_path: str) -> list[dict]:
    try:
        # EasyOCR returns list of [bbox, text, confidence]
        results = reader.readtext(
            image_path,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            canvas_size=1280,   # reduced from 2560 — biggest RAM saver
            mag_ratio=1.0,      # no magnification — saves RAM
        )

        elements = []
        for (bbox, text, confidence) in results:
            if confidence < 0.60:
                logger.info(f'Skipped low conf: "{text}" ({confidence:.2f})')
                continue

            text = text.strip()
            if not text:
                continue

            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # top-left corner
            x = float(bbox[0][0])
            y = float(bbox[0][1])

            elements.append({
                'text': text,
                'x': x,
                'y': y,
                'confidence': round(float(confidence), 3),
            })

            logger.info(f'OCR: "{text}" at ({x:.0f},{y:.0f}) conf:{confidence:.3f}')

        return elements

    except Exception as e:
        logger.error(f"EasyOCR inference error: {e}")
        return []


@app.get('/health')
def health():
    return {'status': 'ok', 'engine': 'easyocr'}


@app.post('/ocr')
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')

    suffix = os.path.splitext(file.filename)[1] if file.filename else '.jpg'

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    tracked_files = [tmp_path]

    try:
        # Preprocess
        preprocessed_path = preprocess_image(tmp_path)
        tracked_files.append(preprocessed_path)

        # Run OCR
        elements = run_ocr(preprocessed_path)

        # Sort by Y position
        elements.sort(key=lambda e: e['y'])

        # Deduplicate elements at same position
        final_elements = []
        seen_keys = set()
        for el in elements:
            pos_key = (round(el['x'], -1), round(el['y'], -1))
            if pos_key not in seen_keys:
                seen_keys.add(pos_key)
                final_elements.append(el)

        gc.collect()

        return {
            'success': True,
            'elements': final_elements,
            'count': len(final_elements),
        }

    except Exception as e:
        logger.error(f"OCR failure: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        for path in tracked_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)