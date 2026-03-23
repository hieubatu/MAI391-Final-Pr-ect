Link File .csv : https://drive.google.com/file/d/1NAnQkTAy0SMxYDlv_X3yvthXCXmdKILW/view?usp=sharing



# 🏡 Hệ thống Dự đoán Giá và Gợi ý Bất động sản bằng Machine Learning

**Dự án học phần MAI391** | **Tác giả:** Nguyễn Đức Hiếu

Dự án ứng dụng Trí tuệ Nhân tạo (Học máy) và các nền tảng Đại số Tuyến tính, Giải tích để giải quyết bài toán định giá bất động sản tự động. Hệ thống không chỉ tính toán mức giá hợp lý dựa trên thông số người dùng nhập vào mà còn hoạt động như một hệ thống gợi ý (Recommendation System), quét thị trường và trả về Top 5 căn nhà thực tế bám sát nhu cầu nhất.

---

## ✨ Tính năng nổi bật

* **🧹 Data Pipeline tự động:** Tự động làm sạch dữ liệu thô, bóc tách diện tích bằng Regex, quy đổi dòng tiền địa phương (Lac, Cr) sang INR và ứng dụng **Z-score** để thanh lọc các điểm dữ liệu dị biệt (Outliers).
* **🤖 Định giá bằng AI (Cập nhật mới nhất):** Tích hợp phép biến đổi **Log Transformation** và thuật toán **Random Forest (Rừng ngẫu nhiên)** với 100 cây quyết định, xử lý xuất sắc tính phi tuyến của thị trường bất động sản phân khúc cao cấp.
* **🔍 Hệ thống Gợi ý (Recommendation):** Áp dụng thuật toán tìm kiếm láng giềng dựa trên **Khoảng cách L1 (Manhattan Distance)** để truy xuất Top 5 căn nhà thực tế có mức giá sát nhất với nhận định của AI.
* **🇻🇳 Bản địa hóa trải nghiệm:** Tự động quy đổi định giá từ đồng Rupee (Ấn Độ) sang Việt Nam Đồng (VNĐ) để người dùng dễ dàng nắm bắt.

---

## 📈 Hiệu suất Mô hình (Model Performance)

Dự án đã trải qua 2 giai đoạn tối ưu hóa thuật toán:
* **Mô hình Cơ sở (Multiple Linear Regression):** Đạt điểm số $R^2 = 0.56$. Hoạt động tốt ở phân khúc tầm trung nhưng gặp sai số (Heteroscedasticity) ở phân khúc cao cấp.
* **Mô hình Tối ưu (Random Forest + Log Transform):** Đạt điểm số **$R^2 = 0.7612$**. Tăng vọt 20% độ chính xác nhờ khả năng bắt chéo các quy luật phức tạp và nén biên độ phương sai.

---

## 🛠 Công nghệ & Thư viện sử dụng

* **Ngôn ngữ:** Python 3.x
* **Xử lý dữ liệu:** `pandas`, `numpy`, `re` (Regular Expression)
* **Machine Learning:** `scikit-learn` (RandomForestRegressor, LinearRegression, train_test_split, metrics)
* **Trực quan hóa:** `matplotlib`, `seaborn`

---

## 📂 Cấu trúc Thư mục

```text
├── house_prices.csv        # Tập dữ liệu gốc (Kaggle)
├── House Price.ipynb       # File Jupyter Notebook (Chứa toàn bộ quá trình EDA, Huấn luyện và Đánh giá)
├── house_price_app.py      # Script chạy trực tiếp trên Terminal/Command Prompt
├── REPORT.pdf              # Báo cáo chuyên sâu về cơ sở toán học của mô hình
└── README.md               # Tài liệu hướng dẫn dự án
