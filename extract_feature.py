import os
import numpy as np
import pickle
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Đường dẫn thư mục và file
DATASET_DIR = 'D:/CBIR/images'  # Thư mục chứa dataset
OUTPUT_DIR = 'D:/CBIR'  # Thư mục chứa file đầu ra
FEATURES_FILE = os.path.join(OUTPUT_DIR, 'features.npy')  # File lưu vector đặc trưng
FILENAMES_FILE = os.path.join(OUTPUT_DIR, 'filenames.pkl')  # File lưu tên các ảnh

# Load EfficientNetB0 pre-trained
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    """Trích xuất vector đặc trưng từ ảnh."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()  # Trả về vector đặc trưng dạng 1D

def process_dataset():
    """Trích xuất đặc trưng từ dataset và lưu vào file."""
    features = []
    filenames = []

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Duyệt qua tất cả ảnh trong dataset
    for filename in os.listdir(DATASET_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Lọc ảnh
            file_path = os.path.join(DATASET_DIR, filename)
            print(f"Đang xử lý: {filename}")
            feature = extract_features(file_path)  # Trích xuất đặc trưng
            features.append(feature)
            filenames.append(filename)

    # Lưu đặc trưng và tên file
    np.save(FEATURES_FILE, np.array(features))  # Lưu vector đặc trưng
    with open(FILENAMES_FILE, 'wb') as f:
        pickle.dump(filenames, f)  # Lưu danh sách tên file

    print(f"Đã lưu {len(features)} đặc trưng vào '{FEATURES_FILE}'")
    print(f"Đã lưu danh sách tên file vào '{FILENAMES_FILE}'")

def check_and_create_files():
    """Kiểm tra và tạo file nếu chưa tồn tại."""
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(FILENAMES_FILE):
        print("File chưa tồn tại. Đang tạo...")
        process_dataset()
    else:
        print("File đặc trưng và danh sách tên file đã tồn tại.")

if __name__ == "__main__":
    check_and_create_files()
