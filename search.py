import numpy as np
import pickle
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# Đường dẫn file
FEATURES_FILE = 'D:/CBIR/features.npy'
FILENAMES_FILE = 'D:/CBIR/filenames.pkl'

# Load EfficientNetB0
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    """Trích xuất vector đặc trưng từ ảnh nhập vào."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

def find_similar_images(query_image_path, top_k=10):
    """Tìm ảnh tương đồng."""
    # Load đặc trưng và tên file
    features = np.load(FEATURES_FILE)
    with open(FILENAMES_FILE, 'rb') as f:
        filenames = pickle.load(f)
    
    # Trích xuất đặc trưng của ảnh truy vấn
    query_features = extract_features(query_image_path)
    
    # Tính cosine similarity
    similarities = cosine_similarity([query_features], features)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]  # Lấy top-k ảnh tương đồng
    
    # Hiển thị kết quả
    print("Ảnh tương đồng nhất:")
    for idx in top_k_indices:
        print(f"{filenames[idx]} - Similarity: {similarities[idx]:.4f}")
    
    # Hiển thị ảnh truy vấn
    plt.figure(figsize=(10, 5))
    query_img = load_img(query_image_path, target_size=(224, 224))
    plt.subplot(top_k + 1, 1, 1)
    plt.imshow(query_img)
    plt.title("Ảnh truy vấn")
    plt.axis('off')
    
    # Hiển thị ảnh tương đồng mỗi ảnh một dòng
    for i, idx in enumerate(top_k_indices):
        image_path = os.path.join('D:/CBIR/images', filenames[idx])
        img = load_img(image_path, target_size=(224, 224))
        plt.subplot(top_k + 1, 1, i + 2)
        plt.imshow(img)
        plt.title(f"{filenames[idx]}\nSimilarity: {similarities[idx]:.4f}")
        plt.axis('off')
    
    plt.tight_layout()  # Đảm bảo các ảnh không bị chồng lên nhau
    plt.show()

if __name__ == "__main__":
    query_image = 'D:/CBIR/images_demo/query_image_11.png'# đường dẫn của ảnh cần tìm kiếm
    find_similar_images(query_image, top_k=5)