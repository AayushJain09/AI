#!/usr/bin/env python3
"""
End-to-End Production Test for Sophisticated Hybrid Recognition System

This script provides comprehensive validation of the production-ready
sophisticated raw + refiner hybrid recognition system using realistic scenarios.
"""

import sys
import numpy as np
import logging
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time

# Add source to path 
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_production_test_items():
    """Create realistic test items similar to items 001, 002, 003."""
    
    def create_realistic_item(item_name, dominant_color, secondary_color, pattern_type):
        """Create realistic item images with variation."""
        items = []
        
        for variation in range(3):  # Create 3 variations per item
            image = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(image)
            
            # Add slight color variation
            color_variation = 20
            varied_dominant = tuple(
                max(0, min(255, c + np.random.randint(-color_variation, color_variation)))
                for c in dominant_color
            )
            varied_secondary = tuple(
                max(0, min(255, c + np.random.randint(-color_variation, color_variation)))
                for c in secondary_color
            )
            
            if pattern_type == 'solid_with_border':
                # Main area
                draw.rectangle([30, 30, 194, 194], fill=varied_dominant, outline=varied_secondary, width=4)
                # Inner details
                draw.rectangle([60, 60, 164, 164], outline=varied_secondary, width=2)
                
            elif pattern_type == 'geometric_pattern':
                # Background
                draw.rectangle([20, 20, 204, 204], fill=varied_dominant)
                # Geometric pattern
                for i in range(3):
                    for j in range(3):
                        x = 50 + i * 50
                        y = 50 + j * 50
                        if (i + j) % 2 == 0:
                            draw.rectangle([x-15, y-15, x+15, y+15], fill=varied_secondary)
                        else:
                            draw.ellipse([x-15, y-15, x+15, y+15], fill=varied_secondary)
                            
            elif pattern_type == 'striped_design':
                # Alternating stripes
                stripe_width = 20
                for i in range(0, 224, stripe_width * 2):
                    draw.rectangle([0, i, 224, i + stripe_width], fill=varied_dominant)
                    if i + stripe_width < 224:
                        draw.rectangle([0, i + stripe_width, 224, i + stripe_width * 2], fill=varied_secondary)
            
            # Add some realistic noise/texture
            for _ in range(10):
                x = np.random.randint(10, 214)
                y = np.random.randint(10, 214)
                size = np.random.randint(1, 3)
                noise_intensity = np.random.randint(-10, 11)
                noise_color = tuple(max(0, min(255, c + noise_intensity)) for c in varied_dominant)
                draw.ellipse([x-size, y-size, x+size, y+size], fill=noise_color)
            
            image_path = f"prod_test_{item_name}_v{variation}.jpg"
            image.save(image_path)
            items.append((image_path, f"{item_name}_v{variation}"))
        
        return items
    
    # Create production-like test items
    all_items = []
    
    # Item 001 equivalent - Red with geometric pattern
    items_001 = create_realistic_item("item_001", (180, 50, 50), (220, 180, 180), 'geometric_pattern')
    all_items.extend(items_001)
    
    # Item 002 equivalent - Blue with solid design  
    items_002 = create_realistic_item("item_002", (50, 50, 180), (150, 150, 220), 'solid_with_border')
    all_items.extend(items_002)
    
    # Item 003 equivalent - Green with striped design
    items_003 = create_realistic_item("item_003", (50, 180, 50), (150, 220, 150), 'striped_design')
    all_items.extend(items_003)
    
    return all_items


def test_production_recognition_system():
    """Test the production-ready sophisticated hybrid recognition system."""
    print("🏭 Production-Ready Sophisticated Hybrid Recognition System Test")
    print("=" * 75)
    
    try:
        # Import production components
        from unified_storage.unified_store import UnifiedStore
        from inference.recognize import UnifiedRecognitionPipeline
        
        # Production configuration (realistic thresholds)
        config = {
            'recognition': {
                'cache_size': 1000,
                'confidence_threshold': 0.85,  # Production threshold
                'high_confidence_threshold': 0.95,  # High confidence
                'min_stage1_confidence': 0.80,  # Stage 1 threshold
                'max_candidate_score_gap': 0.15,  # Gap for rejection
                'min_top_score_margin': 0.03,   # Margin requirement
                'hybrid_mode': True,             # Enable sophisticated hybrid
                'refinement_threshold': 0.82,   # Original refinement threshold
                'confidence_gap_threshold': 0.15,  # Original gap threshold
                # Production ensemble weights
                'ensemble_weights': {
                    'high_raw_confidence': {'raw': 0.85, 'refiner': 0.15},  # Trust raw for high confidence
                    'medium_raw_confidence': {'raw': 0.60, 'refiner': 0.40}, # Balanced for medium
                    'low_raw_confidence': {'raw': 0.30, 'refiner': 0.70}    # Trust refiner for low
                }
            },
            'data': {
                'base_dir': 'production_test_data'
            }
        }
        
        config_path = "production_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize production system
        print("🔧 Initializing production unified storage system...")
        unified_store = UnifiedStore(
            data_dir="production_test_data",
            config_path=config_path,
            enable_analytics=True  # Enable analytics for production
        )
        
        # Create production-ready hybrid pipeline
        print("🔧 Creating production-ready sophisticated hybrid pipeline...")
        pipeline = UnifiedRecognitionPipeline(config['recognition'], unified_store)
        
        print("✅ Production system ready:")
        print(f"  Device: {pipeline.device}")
        print(f"  Hybrid mode: {pipeline.hybrid_mode}")
        print(f"  Refiner available: {pipeline.refiner_model is not None}")
        print(f"  Confidence threshold: {pipeline.thresholds['confidence_threshold']}")
        print(f"  Refinement threshold: {pipeline.refinement_threshold}")
        
        # Create production test dataset
        print("\n📷 Creating production test dataset...")
        test_items = create_production_test_items()
        
        # Separate into reference and query images
        reference_items = [item for item in test_items if item[1].endswith('_v0')]  # v0 as reference
        query_items = [item for item in test_items if not item[1].endswith('_v0')]  # v1, v2 as queries
        
        # Store reference images
        print(f"\n📥 Storing {len(reference_items)} reference images...")
        for image_path, item_id in reference_items:
            category = item_id.split('_v')[0]  # Extract category (item_001, item_002, item_003)
            
            print(f"  Storing reference: {item_id} → {category}")
            
            stored_id = unified_store.store_image(
                image_path=image_path,
                image_id=category,  # Use category as storage ID
                metadata={
                    'category': category,
                    'item_type': 'reference',
                    'production_test': True
                }
            )
        
        print(f"✅ Reference dataset created with {len(reference_items)} items")
        
        # Test recognition on query images
        print(f"\n🔍 Testing production recognition on {len(query_items)} query images...")
        
        results = {
            'item_001': {'correct': 0, 'total': 0, 'raw_only': 0, 'refined': 0, 'times': []},
            'item_002': {'correct': 0, 'total': 0, 'raw_only': 0, 'refined': 0, 'times': []},
            'item_003': {'correct': 0, 'total': 0, 'raw_only': 0, 'refined': 0, 'times': []}
        }
        
        for image_path, item_id in query_items:
            expected_category = item_id.split('_v')[0]  # item_001, item_002, or item_003
            
            print(f"\n🔍 Production test: {item_id}")
            print(f"  Expected: {expected_category}")
            
            # Track hybrid usage before recognition
            pre_raw_only = pipeline.hybrid_stats['raw_only']
            pre_refined = pipeline.hybrid_stats['refined']
            
            # Perform production recognition
            start_time = time.time()
            result = pipeline.recognize_with_unified_storage(image_path)
            recognition_time = time.time() - start_time
            
            # Track hybrid usage
            used_raw_only = pipeline.hybrid_stats['raw_only'] > pre_raw_only
            used_refined = pipeline.hybrid_stats['refined'] > pre_refined
            
            print(f"  Result: {result.item_id} (confidence: {result.confidence:.6f})")
            print(f"  Time: {recognition_time:.3f}s")
            print(f"  Hybrid: {'Raw only' if used_raw_only else 'With refinement'}")
            
            # Show top matches for analysis
            print(f"  Top matches:")
            for i, (match_id, score) in enumerate(result.top_k_matches[:3]):
                print(f"    {i+1}. {match_id}: {score:.6f}")
            
            # Update results
            results[expected_category]['total'] += 1
            results[expected_category]['times'].append(recognition_time)
            
            if used_raw_only:
                results[expected_category]['raw_only'] += 1
            if used_refined:
                results[expected_category]['refined'] += 1
            
            # Check accuracy
            if result.item_id == expected_category:
                results[expected_category]['correct'] += 1
                print(f"  ✅ CORRECT RECOGNITION")
            elif result.item_id == "unknown":
                print(f"  ⚠️  CONSERVATIVE REJECTION (confidence: {result.confidence:.6f})")
            else:
                print(f"  ❌ INCORRECT: {result.item_id}")
        
        # Comprehensive results analysis
        print(f"\n📊 PRODUCTION SYSTEM ANALYSIS")
        print(f"=" * 50)
        
        total_correct = 0
        total_tests = 0
        total_refined = 0
        all_times = []
        
        for item_category, stats in results.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_time = np.mean(stats['times']) if stats['times'] else 0
            refined_pct = (stats['refined'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            print(f"\n{item_category.upper()}:")
            print(f"  Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
            print(f"  Avg time: {avg_time:.3f}s")
            print(f"  Raw only: {stats['raw_only']}")
            print(f"  With refinement: {stats['refined']} ({refined_pct:.1f}%)")
            
            total_correct += stats['correct']
            total_tests += stats['total']
            total_refined += stats['refined']
            all_times.extend(stats['times'])
        
        # Overall system metrics
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
        avg_recognition_time = np.mean(all_times) if all_times else 0
        refinement_usage_pct = (total_refined / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🏆 OVERALL PRODUCTION METRICS:")
        print(f"  Overall accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1%})")
        print(f"  Average recognition time: {avg_recognition_time:.3f}s")
        print(f"  Refinement usage: {total_refined}/{total_tests} ({refinement_usage_pct:.1f}%)")
        
        # System health and performance
        print(f"\n💾 SYSTEM PERFORMANCE:")
        storage_stats = unified_store.get_statistics()
        print(f"  Images stored: {storage_stats.total_images_stored}")
        print(f"  Searches performed: {storage_stats.total_searches_performed}")
        print(f"  Avg storage time: {storage_stats.avg_storage_time_ms:.1f}ms")
        print(f"  Avg search time: {storage_stats.avg_search_time_ms:.1f}ms")
        print(f"  Database size: {storage_stats.database_size_mb:.1f}MB")
        print(f"  System uptime: {storage_stats.uptime_seconds:.1f}s")
        
        # Final hybrid statistics
        print(f"\n🔄 SOPHISTICATED HYBRID ENGINE PERFORMANCE:")
        hybrid_stats = pipeline.get_hybrid_stats()
        print(f"  Total queries: {hybrid_stats['total_queries']}")
        print(f"  Raw only: {hybrid_stats['raw_only']} ({hybrid_stats.get('raw_only_pct', 0):.1f}%)")
        print(f"  With refinement: {hybrid_stats['refined']} ({hybrid_stats.get('refined_pct', 0):.1f}%)")
        
        # Production readiness assessment
        print(f"\n🎯 PRODUCTION READINESS ASSESSMENT:")
        
        production_ready = True
        
        if overall_accuracy < 0.8:
            print(f"  ⚠️  Accuracy below 80%: {overall_accuracy:.1%}")
            production_ready = False
        else:
            print(f"  ✅ Accuracy acceptable: {overall_accuracy:.1%}")
        
        if avg_recognition_time > 1.0:
            print(f"  ⚠️  Recognition time too slow: {avg_recognition_time:.3f}s > 1.0s")
            production_ready = False
        else:
            print(f"  ✅ Recognition time good: {avg_recognition_time:.3f}s")
        
        if storage_stats.success_rate < 0.95:
            print(f"  ⚠️  System reliability low: {storage_stats.success_rate:.1%}")
            production_ready = False
        else:
            print(f"  ✅ System reliability high: {storage_stats.success_rate:.1%}")
        
        # Clean up
        unified_store.close()
        
        # Remove test files
        for image_path, _ in test_items:
            Path(image_path).unlink()
        Path(config_path).unlink()
        
        import shutil
        if Path("production_test_data").exists():
            shutil.rmtree("production_test_data")
        
        if production_ready:
            print(f"\n🎉 PRODUCTION SYSTEM VALIDATION: ✅ PASSED")
            print(f"✅ Sophisticated raw + refiner hybrid system ready for production")
            print(f"✅ Unified storage backend fully functional")
            print(f"✅ Recognition accuracy and performance meet requirements")
            print(f"✅ System reliability and health metrics acceptable")
        else:
            print(f"\n⚠️  PRODUCTION SYSTEM VALIDATION: Needs attention")
        
        return production_ready
        
    except Exception as e:
        logger.error(f"❌ Production system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the production-ready system test."""
    success = test_production_recognition_system()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())