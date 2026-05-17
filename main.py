import os
import gc
import logging
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimization: Limit PaddlePaddle internal memory pre-allocation
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0' # Ensure no GPU attempt

app = FastAPI(title="EF Analyzer OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Global OCR Instance - Optimized for low-RAM environments (Railway/HF)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    show_log=False,
    ocr_version='PP-OCRv4',
    rec_batch_num=1,          # CRITICAL: Processes 1 line at a time to save RAM
    use_mp=False,             # Disable multi-processing to keep memory footprint flat
    total_process_num=1,
    enable_mkldnn=False,      # Disabled to prevent extra memory overhead
)

def process_and_segment_image(input_path: str) -> list[tuple[str, int]]:
    """
    Handles resizing, enhancement, and cropping in a memory-efficient flow.
    Returns a list of (temp_file_path, y_offset).
    """
    segment_paths = []
    
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        
        # 1. Targeted Upscaling
        if w < 1080:
            scale = 1080 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size
            logger.info(f"Upscaled to {w}x{h}")

        # 2. Enhancement (In-place where possible)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Sharpness(img).enhance(1.5)

        # 3. Dynamic Cropping
        # We split the image to help PaddleOCR focus and prevent OOM on massive files
        regions = [
            ((0, 0, w, int(h * 0.22)), 0, 'header'),       # Top 22%
            ((0, int(h * 0.20), w, h), int(h * 0.20), 'stats') # Remaining 80% (slight overlap)
        ]

        for box, y_offset, label in regions:
            with img.crop(box) as chunk:
                # Use JPEG to reduce disk I/O and intermediate memory pressure
                chunk_path = f"{input_path}_{label}.jpg"
                chunk.save(chunk_path, 'JPEG', quality=90)
                segment_paths.append((chunk_path, y_offset))
                
    # Explicitly clear image objects from memory
    gc.collect()
    return segment_paths

def run_ocr_on_file(file_path: str) -> list[dict]:
    """Runs inference on a single file chunk."""
    try:
        result = ocr.ocr(file_path, cls=True)
        elements = []

        if result and result[0]:
            for line in result[0]:
                box, (text, confidence) = line
                if confidence < 0.60:
                    continue

                elements.append({
                    'text': text.strip(),
                    'x': float(box[0][0]),
                    'y': float(box[0][1]),
                    'confidence': round(float(confidence), 3),
                })
        return elements
    except Exception as e:
        logger.error(f"Inference error on {file_path}: {e}")
        return []

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ocr')
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')

    # Use a unique temp file for the original upload
    suffix = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        main_tmp_path = tmp.name

    tracked_files = [main_tmp_path]
    all_elements = []

    try:
        # Step 1: Preprocess and generate chunks
        segments = process_and_segment_image(main_tmp_path)
        
        # Step 2: OCR each segment
        for segment_path, y_offset in segments:
            tracked_files.append(segment_path)
            
            chunk_results = run_ocr_on_file(segment_path)
            for el in chunk_results:
                el['y'] += y_offset # Normalize Y coordinate
                all_elements.append(el)
            
            # Clear memory after each segment processing
            gc.collect()

        # Step 3: Global Sorting and Deduplication
        all_elements.sort(key=lambda e: e['y'])
        
        final_elements = []
        seen_keys = set()
        for el in all_elements:
            # Snap to grid to find overlaps (10px tolerance)
            pos_key = (round(el['x'], -1), round(el['y'], -1))
            if pos_key not in seen_keys:
                seen_keys.add(pos_key)
                final_elements.append(el)

        return {
            'success': True,
            'elements': final_elements,
            'count': len(final_elements)
        }

    except Exception as e:
        logger.error(f"Global OCR failure: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup all temporary artifacts
        for path in tracked_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        gc.collect()

if __name__ == '__main__':
    # Default to Railway's port 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)