from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
import uvicorn
import tempfile
import os
import logging

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
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    show_log=False,
    ocr_version='PP-OCRv4',
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.8,
    rec_batch_num=6,
    max_text_length=25,
    use_space_char=True,
)


def preprocess_image(image_path: str) -> str:
    img = Image.open(image_path).convert('RGB')

    # Step 1 — upscale if too small
    w, h = img.size
    if w < 1080:
        scale = 1080 / w
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS
        )
        logger.info(f'Upscaled from {w}x{h} to {img.size}')

    # Step 2 — boost contrast
    img = ImageEnhance.Contrast(img).enhance(1.5)

    # Step 3 — boost sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Save preprocessed image
    preprocessed_path = image_path + '_processed.png'
    img.save(preprocessed_path, 'PNG')

    return preprocessed_path


def crop_regions(image_path: str) -> list[tuple[str, int]]:
    """
    Crop image into regions and return list of (path, y_offset) tuples.
    Splitting header from stats table improves accuracy per region.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    regions = [
        # (crop_box, y_offset, name)
        ((0, 0, w, int(h * 0.20)), 0, 'header'),
        ((0, int(h * 0.20), w, int(h * 0.90)), int(h * 0.20), 'stats'),
    ]

    saved = []
    for box, y_offset, name in regions:
        cropped = img.crop(box)
        path = image_path + f'_region_{name}.png'
        cropped.save(path)
        saved.append((path, y_offset))

    return saved


def run_ocr_on_file(file_path: str) -> list[dict]:
    result = ocr.ocr(file_path, cls=True)
    elements = []

    if result and result[0]:
        for line in result[0]:
            box, (text, confidence) = line

            if confidence < 0.65:
                logger.info(f'Skipped low confidence: "{text}" ({confidence:.2f})')
                continue

            text = text.strip()
            if not text:
                continue

            x = float(box[0][0])
            y = float(box[0][1])

            elements.append({
                'text': text,
                'x': x,
                'y': y,
                'confidence': round(confidence, 3),
            })

    return elements


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/ocr')
async def recognize(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')

    # Determine suffix
    suffix = '.png' if 'png' in (file.content_type or '') else '.jpg'

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    temp_files = [tmp_path]
    preprocessed_path = None

    try:
        # Step 1 — preprocess
        preprocessed_path = preprocess_image(tmp_path)
        temp_files.append(preprocessed_path)

        # Step 2 — crop into regions
        regions = crop_regions(preprocessed_path)
        for region_path, _ in regions:
            temp_files.append(region_path)

        # Step 3 — OCR each region
        all_elements = []

        for region_path, y_offset in regions:
            region_elements = run_ocr_on_file(region_path)

            for el in region_elements:
                # Adjust Y coordinate back to full image space
                el['y'] = el['y'] + y_offset
                all_elements.append(el)

                logger.info(
                    f'OCR: "{el["text"]}" '
                    f'at ({el["x"]:.0f}, {el["y"]:.0f}) '
                    f'conf:{el["confidence"]}'
                )

        # Step 4 — sort by Y position top to bottom
        all_elements.sort(key=lambda e: e['y'])

        # Step 5 — deduplicate elements at same position
        # (can happen at region boundaries)
        deduplicated = []
        seen_positions = set()

        for el in all_elements:
            key = (round(el['x'], -1), round(el['y'], -1))
            if key not in seen_positions:
                seen_positions.add(key)
                deduplicated.append(el)

        return {
            'success': True,
            'elements': deduplicated,
            'count': len(deduplicated),
        }

    except Exception as e:
        logger.error(f'OCR error: {e}')
        raise HTTPException(500, f'OCR processing failed: {str(e)}')

    finally:
        # Clean up all temp files
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info',
    )