from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # Để sử dụng điều kiện chờ
import requests
import os

# Đường dẫn đến ChromeDriver
chrome_driver_path = 'd:/chromedriver-win64/chromedriver.exe'
service = Service(chrome_driver_path)

# Tạo trình duyệt Selenium
driver = webdriver.Chrome(service=service)

# URL của trang web cần crawl
url = "https://www.thegioididong.com/dtdd#c=42&o=13&pi=5"  # Thay bằng URL của bạn

try:
    # Mở trang web
    driver.get(url)
    
    # Đợi trang web tải xong và tìm tất cả thẻ <img> bằng WebDriverWait
    try:
        img_elements = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.item-img img.thumb"))
        )
    except Exception as e:
        print("Không tìm thấy ảnh trong trang web:", e)
        driver.quit()
        exit()

    # Thư mục lưu ảnh
    save_directory = "C:/Users/Tien/Documents/CBIR/images"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Lặp qua tất cả các thẻ ảnh và tải về
    for idx, img_element in enumerate(img_elements):
        # Lấy URL ảnh từ `data-src` hoặc `src`
        img_url = img_element.get_attribute("data-src") or img_element.get_attribute("src")
        if img_url:
            print(f"Tìm thấy URL ảnh {idx + 1}: {img_url}")
            
            # Tải ảnh về
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                # Lưu ảnh
                img_name = os.path.basename(img_url)
                img_path = os.path.join(save_directory, img_name)
                with open(img_path, "wb") as file:
                    file.write(img_response.content)
                print(f"Đã tải ảnh {idx + 1}: {img_path}")
            else:
                print(f"Không thể tải ảnh {idx + 1} từ URL: {img_url}")
        else:
            print(f"Không tìm thấy URL ảnh cho ảnh thứ {idx + 1}.")
finally:
    # Đóng trình duyệt
    driver.quit()