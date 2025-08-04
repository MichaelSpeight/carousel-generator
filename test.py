from PIL import Image, ImageStat, ImageDraw
from ultralytics import YOLO
import numpy as np

def detect_phones(image_path):
    model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model
    results = model(image_path)
    boxes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            if label.lower() == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    return boxes

def box_overlap(box1, box2):
    """Returns True if box1 overlaps with box2"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

def find_best_text_region(image, avoid_boxes=[], box_size=(400, 300), stride=100):
    gray = image.convert("L")
    width, height = image.size
    best_score = float('inf')
    best_box = (0, 0)

    for y in range(0, height - box_size[1], stride):
        for x in range(0, width - box_size[0], stride):
            candidate_box = (x, y, x + box_size[0], y + box_size[1])

            # Skip if overlaps with any avoid box (e.g., phone)
            if any(box_overlap(candidate_box, b) for b in avoid_boxes):
                continue

            region = gray.crop(candidate_box)
            stat = ImageStat.Stat(region)
            contrast = stat.stddev[0]
            brightness = stat.mean[0]

            score = contrast + abs(brightness - 128)
            if score < best_score:
                best_score = score
                best_box = (x, y)

    return best_box + (best_box[0] + box_size[0], best_box[1] + box_size[1])

if __name__ == "__main__":
    input_image = "test_1.png"
    output_image = "test_output.jpg"
    box_size = (400, 300)

    img = Image.open(input_image).convert("RGB")
    detected_boxes = detect_phones(input_image)
    print(f"📱 Detected phones: {detected_boxes}")

    best_box = find_best_text_region(img, avoid_boxes=detected_boxes, box_size=box_size)

    # Draw result
    draw = ImageDraw.Draw(img)
    draw.rectangle(best_box, outline="magenta", width=5)
    for box in detected_boxes:
        draw.rectangle(box, outline="red", width=3)

    img.save(output_image)
    print(f"✅ Saved output with text box at {output_image}")


# from PIL import Image, ImageStat, ImageDraw

# def find_best_text_region(image, box_size=(400, 300), stride=100):
#     gray = image.convert("L")  # grayscale
#     width, height = image.size
#     best_score = float('inf')
#     best_box = (0, 0)

#     for y in range(0, height - box_size[1], stride):
#         for x in range(0, width - box_size[0], stride):
#             box = (x, y, x + box_size[0], y + box_size[1])
#             region = gray.crop(box)
#             stat = ImageStat.Stat(region)
#             contrast = stat.stddev[0]
#             brightness = stat.mean[0]

#             # Prioritize low contrast, mid-tone brightness
#             score = contrast + abs(brightness - 128)
#             if score < best_score:
#                 best_score = score
#                 best_box = (x, y)

#     return best_box + (best_box[0] + box_size[0], best_box[1] + box_size[1])  # return as full box

# if __name__ == "__main__":
#     # input_image = "test.jpg"
#     input_image = "test_1.png"
#     output_image = "test_output.jpg"
#     box_size = (400, 300)

#     img = Image.open(input_image).convert("RGB")
#     best_box = find_best_text_region(img, box_size=box_size)

#     # Draw rectangle
#     draw = ImageDraw.Draw(img)
#     draw.rectangle(best_box, outline="magenta", width=5)

#     img.save(output_image)
#     print(f"✅ Saved output with box at {output_image}")
