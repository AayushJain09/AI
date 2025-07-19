from fastapi import FastAPI
import sys
sys.path.append('src')
from inference.recognize import create_pipeline

app = FastAPI()
pipeline = create_pipeline('config.yaml')

@app.get("/")
def health():
    return {"status": "ok", "vectors": pipeline.index.ntotal, "dimensions": pipeline.index.d}

@app.get("/test/{item_id}")
def test_item(item_id: str):
    test_image = f'data/raw/{item_id}/1752950980813_IMG_8416.JPG'
    result = pipeline.recognize(test_image)
    return {"item": result.item_id, "confidence": result.confidence}

@app.get("/test_unknown")
def test_unknown():
    test_image = './environments/env2/lib/python3.13/site-packages/networkx/drawing/tests/baseline/test_display_empty_graph.png'
    result = pipeline.recognize(test_image)
    return {"item": result.item_id, "confidence": result.confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)