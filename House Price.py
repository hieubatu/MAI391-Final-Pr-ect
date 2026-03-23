import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# 1. HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def extract_number(val):
    if pd.isna(val): return np.nan
    nums = re.findall(r"[\d\.]+", str(val).replace(',', ''))
    if nums:
        try: return float(nums[0])
        except: pass
    return np.nan

def parse_amount(val):
    if pd.isna(val): return np.nan
    val_str = str(val).lower().replace(',', '')
    num = extract_number(val_str)
    if pd.isna(num): return np.nan
    if 'lac' in val_str: return num * 100000        
    elif 'cr' in val_str: return num * 10000000     
    else: return num

# ==========================================
# 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
print("Đang tải và làm sạch dữ liệu từ house_prices.csv...")
df = pd.read_csv('house_prices.csv')

df['Price_Rupees'] = df['Amount(in rupees)'].apply(parse_amount)
df['Area_Sqft'] = df['Carpet Area'].apply(extract_number)
df['Baths'] = df['Bathroom'].apply(extract_number)

df_clean = df[['Price_Rupees', 'Area_Sqft', 'Baths']].dropna()
df_clean = df_clean[(df_clean['Area_Sqft'] < 10000) & (df_clean['Price_Rupees'] < 500000000)]
print(f"Đã dọn dẹp xong! Còn lại {df_clean.shape[0]} căn nhà hợp lệ.\n")

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH LINEAR REGRESSION
# ==========================================
X = df_clean[['Area_Sqft', 'Baths']]
y = df_clean['Price_Rupees']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f" Điểm số R-squared của mô hình: {r2_score(y_test, y_pred):.2f}\n")

# ==========================================
# 4. HỆ THỐNG GỢI Ý & ĐỊNH GIÁ TƯƠNG TÁC
# ==========================================
TY_GIA = 305 # 1 INR = 305 VND

print("="*60)
print("       HỆ THỐNG GỢI Ý BẤT ĐỘNG SẢN BẰNG AI")
print("="*60)

try:
    dien_tich_input = float(input("Nhập Diện tích bạn muốn (sqft): "))
    phong_tam_input = float(input("Nhập Số phòng tắm: "))

    # AI định giá
    nha_moi = pd.DataFrame({'Area_Sqft': [dien_tich_input], 'Baths': [phong_tam_input]})
    gia_du_doan_inr = model.predict(nha_moi)[0]
    gia_du_doan_vnd = gia_du_doan_inr * TY_GIA

    print(f"\nCĂN NHÀ NÀY TRỊ GIÁ KHOẢNG: {gia_du_doan_inr:,.0f} INR (~ {gia_du_doan_vnd:,.0f} VNĐ)")

    # Tìm 5 căn nhà khớp tiêu chí
    df_cung_phong_tam = df_clean[df_clean['Baths'] == phong_tam_input].copy()

    if df_cung_phong_tam.empty:
        print("\nKhông tìm thấy nhà có đúng số phòng tắm này, đang mở rộng tìm kiếm...")
        df_cung_phong_tam = df_clean.copy()

    df_cung_phong_tam['Độ lệch (INR)'] = abs(df_cung_phong_tam['Price_Rupees'] - gia_du_doan_inr)
    top_5 = df_cung_phong_tam.sort_values(by='Độ lệch (INR)').head(5).copy()

    # Định dạng bảng xuất ra terminal
    top_5['Giá Thực tế (VNĐ)'] = top_5['Price_Rupees'] * TY_GIA
    bang_ket_qua = top_5[['Area_Sqft', 'Baths', 'Price_Rupees', 'Giá Thực tế (VNĐ)', 'Độ lệch (INR)']]
    bang_ket_qua.columns = ['Diện tích (sqft)', 'Phòng tắm', 'Giá Thực tế (INR)', 'Giá Thực tế (VNĐ)', 'Độ lệch so với AI (INR)']

    pd.options.display.float_format = '{:,.0f}'.format
    print("\nĐÂY LÀ BẢNG TOP 5 CĂN NHÀ ĐANG BÁN THỎA MÃN TIÊU CHÍ CỦA BẠN:\n")
    print(bang_ket_qua.to_string(index=False))
    print("\n" + "="*60)

except ValueError:
    print("\nLỗi: Vui lòng nhập một con số hợp lệ!")

# ==========================================
# 5. VẼ BIỂU ĐỒ TRỰC QUAN
# ==========================================
print("\nĐang mở biểu đồ phân tích... (Tắt cửa sổ biểu đồ để kết thúc chương trình)")
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá Thực Tế (INR)')
plt.ylabel('Giá AI Dự Đoán (INR)')
plt.title('Hồi quy tuyến tính: Thực tế vs Dự đoán')
plt.grid(True)
plt.show()