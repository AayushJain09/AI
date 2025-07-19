#\!/usr/bin/env python3
import sys
sys.path.append('src')
from inference.recognize import create_pipeline

print("Testing recognition pipeline that backend should use...")

pipeline = create_pipeline('config.yaml')

# Test with known item
result = pipeline.recognize('data/raw/item_004/1752950980813_IMG_8416.JPG')
print(f"Known item test: {result.item_id} (confidence: {result.confidence:.3f})")

# Test with unknown item
result = pipeline.recognize('./environments/env2/lib/python3.13/site-packages/networkx/drawing/tests/baseline/test_display_empty_graph.png')
print(f"Unknown item test: {result.item_id} (confidence: {result.confidence:.3f})")

print("Pipeline test completed successfully\!")
