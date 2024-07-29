
import os
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import multiprocessing


def detect_and_crop_face(img, detector):
    """Detects the largest face in an image using MTCNN and crops it."""
    if img is None:
        return None  # Ensure the image is read correctly

    # Convert image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = detector.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        return None  # No faces found

    # Find the largest face
    largest_face_index = max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    x, y, x2, y2 = boxes[largest_face_index]

    # Crop the face
    face = img_rgb[int(y):int(y2), int(x):int(x2)]

    # Convert the cropped face back to PIL image
    return Image.fromarray(face)

def process_image(filename, input_folder, output_folder, detector):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        base_name, ext = os.path.splitext(filename)
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)
        face_image = detect_and_crop_face(img, detector)
        if face_image is not None:
            output_filename = f"{base_name}_cropped{ext}"
            output_path = os.path.join(output_folder, output_filename)
            face_image.save(output_path)
        else:
            print(f"No face detected in {filename}.")

def process_images_concurrently(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize MTCNN detector once
    detector = MTCNN()

    # Create a list of all file names
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Setup multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    tasks = [(filename, input_folder, output_folder, detector) for filename in filenames]
    pool.starmap(process_image, tasks)
    pool.close()
    pool.join()

# Define your folders
input_folder = r'/Users/lap01743/Downloads/WorkSpace/capstone_wed/XMT_Model/6k_fake_styleGan1'
output_folder = r'/Users/lap01743/Downloads/WorkSpace/capstone_wed/XMT_Model/35k_stylegan1_images_rm_background'

# Process the images concurrently
if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_images_concurrently(input_folder, output_folder)
