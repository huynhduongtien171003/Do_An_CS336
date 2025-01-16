from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pickle
import os
import cv2  # Thư viện xử lý ảnh OpenCV

# Tải mô hình VGG16
def load_model():
    base_model = VGG16(weights="imagenet", include_top=False)
    model = Model(inputs=base_model.inputs, outputs=base_model.output)
    return model

# Hàm trích xuất đặc trưng từ VGG16
def extract_vgg16_features(image: Image.Image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Hàm trích xuất đặc trưng góc cạnh (Edge Features)
def extract_edge_features(image: Image.Image):
    if image.mode != "L":
        image = image.convert("L")  # Chuyển đổi ảnh sang grayscale
    image_array = np.array(image)
    # Sử dụng Sobel để phát hiện biên cạnh
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    # Chuẩn hóa đặc trưng biên cạnh
    sobel_flattened = sobel_combined.flatten().astype("float32")
    if np.linalg.norm(sobel_flattened) > 0:
        sobel_flattened /= np.linalg.norm(sobel_flattened)
    return sobel_flattened

# Hàm chuẩn hóa đặc trưng
def normalize_feature(feature, target_length):
    """
    Chuẩn hóa đặc trưng về chiều dài target_length.
    """
    if len(feature) > target_length:
        return feature[:target_length]
    padded_feature = np.zeros(target_length, dtype="float32")
    padded_feature[:len(feature)] = feature
    return padded_feature

# Hàm kết hợp tất cả đặc trưng
def combine_features(vgg16_features, edge_features, vgg_weight=0.7, edge_weight=0.3):
    vgg16_features_scaled = vgg16_features * vgg_weight
    edge_features_scaled = edge_features * edge_weight
    return np.concatenate((vgg16_features_scaled, edge_features_scaled))

# Hàm lưu đặc trưng vào file .pkl
def save_features(features, paths, feature_file="features.pkl"):
    with open(feature_file, "wb") as f:
        pickle.dump({"features": features, "paths": paths}, f)

# Hàm khởi tạo và lưu đặc trưng
def initialize_features(image_dir="D:/CBIR/images", feature_file="D:/CBIR/features.pkl"):
    if not os.path.exists(image_dir):
        print(f"Thư mục {image_dir} không tồn tại. Vui lòng tạo và thêm ảnh.")
        return

    features, paths = [], []
    max_allowed_length = 1000000  # Giới hạn chiều dài đặc trưng tối đa
    target_length = 500000  # Chiều dài đặc trưng cố định sau chuẩn hóa

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path)
            # Trích xuất đặc trưng
            vgg16_features = extract_vgg16_features(image, model)
            edge_features = extract_edge_features(image)

            # Kết hợp đặc trưng với tỷ lệ phù hợp
            combined_feature = combine_features(vgg16_features, edge_features, vgg_weight=0.7, edge_weight=0.3)

            # Kiểm tra chiều dài đặc trưng
            if len(combined_feature) > max_allowed_length:
                print(f"Bỏ qua ảnh {image_name} vì đặc trưng quá dài ({len(combined_feature)}).")
                continue

            features.append(combined_feature)
            paths.append(image_path)
            print(f"Đã xử lý ảnh: {image_name}")

        except Exception as e:
            print(f"Lỗi xử lý ảnh {image_name}: {e}")

    # Chuẩn hóa tất cả đặc trưng về chiều dài cố định
    features = [normalize_feature(f, target_length) for f in features]

    # Lưu đặc trưng vào file
    save_features(features, paths, feature_file)
    print(f"Đặc trưng đã được trích xuất và lưu vào {feature_file}")

# Tải mô hình
model = load_model()

# Khởi tạo và lưu đặc trưng
if __name__ == "__main__":
    initialize_features()
