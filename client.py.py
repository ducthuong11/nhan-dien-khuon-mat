import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ file CSV
file_path = "D:/KY2/AIoT/BTL/sensor_data_activity.csv" 
data = pd.read_csv(file_path)

# 1. Làm sạch dữ liệu
data = data.dropna()  # Xóa giá trị bị thiếu
data = data.drop_duplicates()  # Xóa dòng trùng lặp

# 2. Lọc hoạt động hợp lệ
valid_activities = ["fall_forward", "fall_backward", "fall_left", "fall_right", "standing"]
data = data[data["ActivityLabel"].isin(valid_activities)]

# 3. Mã hóa nhãn thành số
label_encoder = LabelEncoder()
data["ActivityLabel"] = label_encoder.fit_transform(data["ActivityLabel"])

# 4. Vẽ biểu đồ số lượng mẫu theo từng hoạt động
plt.figure(figsize=(8, 5))
activity_counts = data["ActivityLabel"].value_counts()
activity_names = label_encoder.inverse_transform(activity_counts.index)  # Chuyển nhãn số về tên gốc

plt.bar(activity_names, activity_counts.values, color="steelblue")

# Cấu hình biểu đồ
plt.xlabel("Activity")
plt.ylabel("Number of Samples")
plt.title("Number of Samples per Activity")
plt.xticks(rotation=30)  # Xoay nhãn trục x để dễ đọc

# Hiển thị biểu đồ
plt.show()

# 5. Tạo cửa sổ trượt
window_size = 50  # 50 mẫu
step_size = 25    # Trượt 25 mẫu (50% overlap)

windows = []
labels = []

for i in range(0, len(data) - window_size, step_size):
    window = data.iloc[i:i + window_size, 1:-1].values  # Lấy dữ liệu cảm biến (bỏ cột "Time")
    label = data.iloc[i + window_size - 1]["ActivityLabel"]  # Lấy nhãn tại mẫu cuối của cửa sổ
    windows.append(window)
    labels.append(label)

# Chuyển thành numpy array
X = np.array(windows)  # Dữ liệu cảm biến (shape: num_windows x 50 x 6)
y = np.array(labels)   # Nhãn tương ứng (shape: num_windows)

# Lưu dữ liệu đã xử lý
save_path = "D:/KY2/AIoT/BTL/"
np.save(save_path + "X_windows.npy", X)
np.save(save_path + "y_labels.npy", y)

print(f"Số lượng cửa sổ trượt: {len(X)}")
print(f"Shape của X: {X.shape} (số cửa sổ, 50 mẫu, 6 đặc trưng)")
print(f"Shape của y: {y.shape} (số cửa sổ)")
