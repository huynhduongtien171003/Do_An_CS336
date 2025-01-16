import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import cv2

# Đường dẫn file đặc trưng
FEATURES_FILE = 'D:/CBIR/features.npy'
FILENAMES_FILE = 'D:/CBIR/filenames.pkl'

# Load EfficientNetB0 model
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# Load đặc trưng và tên file
if os.path.exists(FEATURES_FILE) and os.path.exists(FILENAMES_FILE):
    features = np.load(FEATURES_FILE)
    with open(FILENAMES_FILE, 'rb') as f:
        filenames = pickle.load(f)
else:
    st.error("File đặc trưng hoặc danh sách tên file không tồn tại.")
    st.stop()

def extract_features(image):
    """Trích xuất vector đặc trưng từ ảnh."""
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

def find_similar_images(query_features, top_k=10):
    """Tìm top-k ảnh tương đồng."""
    similarities = cosine_similarity([query_features], features)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = [{"filename": filenames[idx], "similarity": similarities[idx]} for idx in top_k_indices]
    return results

def blur_background(image):
    """Làm mờ nền và giữ lại đối tượng chính."""
    # Chuyển ảnh sang không gian màu HSV để dễ dàng tạo mặt nạ
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu để tạo mặt nạ (ví dụ: màu da hoặc màu đồng hồ)
    lower_bound = np.array([0, 0, 0])  # Thay đổi giá trị này để phù hợp với đối tượng
    upper_bound = np.array([180, 255, 255])  # Thay đổi giá trị này để phù hợp với đối tượng

    # Tạo mặt nạ
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Làm mờ nền
    blurred_background = cv2.GaussianBlur(image, (51, 51), 0)

    # Kết hợp ảnh gốc và nền đã làm mờ
    result = cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(blurred_background, blurred_background, mask=~mask)

    return result

# Giao diện Streamlit
st.title("Tìm kiếm ảnh tương đồng")
st.write("Tải ảnh của bạn lên và hệ thống sẽ trả về 10 ảnh tương đồng nhất.")

uploaded_file = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh tải lên
    query_image = Image.open(uploaded_file).convert("RGB")
    query_image_cv = np.array(query_image)  # Chuyển đổi PIL Image sang numpy array
    query_image_cv = cv2.cvtColor(query_image_cv, cv2.COLOR_RGB2BGR)  # Chuyển đổi từ RGB sang BGR

    # Làm mờ nền
    query_image_blurred = blur_background(query_image_cv)

    # Hiển thị ảnh đã xử lý
    st.image(cv2.cvtColor(query_image_blurred, cv2.COLOR_BGR2RGB), caption="Ảnh đã xử lý", use_column_width=True)

    # Tìm kiếm ảnh tương đồng
    with st.spinner("Đang tìm kiếm ảnh tương đồng..."):
        query_features = extract_features(Image.fromarray(cv2.cvtColor(query_image_blurred, cv2.COLOR_BGR2RGB)))
        results = find_similar_images(query_features)

    # Hiển thị kết quả
    st.write("### 10 ảnh tương đồng nhất:")
    cols = st.columns(3)  # Hiển thị 3 cột để ảnh lớn hơn
    for i, result in enumerate(results):
        similar_image_path = os.path.join('D:/CBIR/images', result["filename"])
        similar_image = Image.open(similar_image_path).convert("RGB").resize((200, 200))  # Tăng kích thước ảnh
        
        # Bỏ đuôi file ảnh (ví dụ: .jpg)
        image_name = os.path.splitext(result["filename"])[0]
        
        cols[i % 3].image(similar_image, caption=image_name, use_column_width=True)  # Hiển thị tên ảnh (không có đuôi)
        st.markdown("---")  # Thêm đường kẻ ngang giữa các ảnh