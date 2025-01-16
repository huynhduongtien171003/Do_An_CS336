from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time

# Đường dẫn đến ChromeDriver
chrome_driver_path = 'd:/chromedriver-win64/chromedriver.exe'
service = Service(chrome_driver_path)

# Cấu hình ChromeOptions
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")  # Chạy ở chế độ không có giao diện (nếu cần)
chrome_options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"

# Tạo trình duyệt Selenium
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL của trang web cần crawl
url = "https://empireluxury.vn/thuong-hieu/dong-ho-rolex/?srsltid=AfmBOorqK1kYJsD0F_E99p7JfOV05lU_Lg2k93_RleEJ87gQRFgFMPEv"

try:
    # Mở trang web
    driver.get(url)

    # Cuộn trang xuống để tải nội dung lazy loading
    for _ in range(5):  # Thử cuộn 5 lần
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Tăng thời gian chờ

    # Đợi và tìm các sản phẩm đồng hồ
    try:
        product_elements = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.col-md-3.mb-3"))
        )
    except Exception as e:
        print("Không tìm thấy sản phẩm trên trang web:", e)
        driver.quit()
        exit()

    # Thư mục lưu ảnh
    save_directory = "D:/CBIR/images_watch"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Lặp qua tất cả các sản phẩm
    for idx, product_element in enumerate(product_elements):
        try:
            # Tìm tên đồng hồ từ thuộc tính title của thẻ <a>
            name_element = product_element.find_element(By.CSS_SELECTOR, "a.thumbnail.zoom-on-hover")
            watch_name = name_element.get_attribute("title").strip()

            # Tìm URL ảnh từ thẻ <img>
            img_element = product_element.find_element(By.CSS_SELECTOR, "img.img-fluid.wp-post-image")
            img_url = img_element.get_attribute("src")

            if img_url:
                print(f"Tìm thấy URL ảnh {idx + 1}: {img_url}")

                # Tải ảnh về
                try:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        # Đặt tên ảnh theo tên đồng hồ
                        img_name = f"{watch_name.replace(' ', '_')}.jpg"
                        img_path = os.path.join(save_directory, img_name)
                        with open(img_path, "wb") as file:
                            file.write(img_response.content)
                        print(f"Đã tải ảnh {idx + 1}: {img_path}")
                    else:
                        print(f"Không thể tải ảnh {idx + 1} từ URL: {img_url}")
                except requests.exceptions.RequestException as e:
                    print(f"Lỗi khi tải ảnh {idx + 1}: {e}")
            else:
                print(f"Không tìm thấy URL ảnh cho sản phẩm thứ {idx + 1}.")
        except Exception as e:
            print(f"Lỗi khi xử lý sản phẩm thứ {idx + 1}: {e}")

finally:
    # Đóng trình duyệt
    driver.quit()