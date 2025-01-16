# CS336 - CONTENT BASED IMAGE RETRIEVAL

## My Team
| Name               | MSSV        |
|--------------------|-------------|
| Huỳnh Dương Tiến   | 22521465    |
| Huỳnh Nhật Minh    | 22520862    |
| Đoàn Văn Hoàng     | 22521465    |

## Architecture Code

### `File crawl.py`
- Dùng để crawl dữ liệu từ website **thegioididong.com** để lấy ảnh các thiết bị điện tử.

### `File crawl_dongho.py`
- Dùng để crawl dữ liệu từ các web bán đồng hồ để lấy ảnh đồng hồ.

### `File extract_feature.py`
- Trích xuất đặc trưng từ tất cả các ảnh trong thư mục **images** bằng **EfficientNetB0**.

### `File extract_features.py`
- Trích xuất đặc trưng từ tất cả các ảnh trong thư mục **images**, kết hợp giữa **VGG**, góc cạnh và màu sắc.

### `Folder images`
- Chứa các ảnh dùng để tìm kiếm ảnh tương đồng.

### `Folder images_demo`
- Chứa các ảnh để chạy demo và test tìm kiếm ảnh tương đồng.

### `File Search.py`
- Dùng để tìm kiếm ảnh tương đồng trong thư mục **images**.

### `File APP.py`
- Dùng để chạy **API** của mô hình.

## Usage

### Chạy `Search.py`:
1. **Sửa đường dẫn các file đặc trưng đã trích xuất** và **đường dẫn chứa ảnh cần tìm kiếm** .
   
   #### Đường dẫn file:
   ```python
   FEATURES_FILE = 'D:/CBIR/features.npy'
   FILENAMES_FILE = 'D:/CBIR/filenames.pkl'
   query_image = 'D:/CBIR/images_demo/query_image_11.png' # Đường dẫn của ảnh cần tìm kiếm
### Chạy `APP.py`:
1. **Sửa đường dẫn các file đặc trưng đã trích xuất**  .
   
   #### Đường dẫn file:
   ```python
   FEATURES_FILE = 'D:/CBIR/features.npy'
   FILENAMES_FILE = 'D:/CBIR/filenames.pkl'
Lưu ý: Sửa lại thành đường dẫn trong máy của bạn để chạy.

2. **Sửa dụng lệnh**  để chạy file `APP.py`.
   ```python
   Streamlit run APP.py
## Link chứa các file đặc trưng đã trích xuất bằng EfficientNetB0: https://drive.google.com/drive/folders/1Qp-Ivxe4wnyZShe-xju9EzJqbqANt6XR?usp=drive_link
