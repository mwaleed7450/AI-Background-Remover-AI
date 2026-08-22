from quality_analysis import analyze_image_quality
import json

if __name__ == "__main__":
    result = analyze_image_quality("test_images/dog_small.jpeg")
    print(json.dumps(result, indent=2))