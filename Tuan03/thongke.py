import pandas as pd
import numpy as np

# Tập dữ liệu tuổi
age_data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 
            30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]

# 1. Tính Q1, Q3, Mean, Median
mean_val = np.mean(age_data)
median_val = np.median(age_data)
q1 = np.percentile(age_data, 25, method='midpoint') # Tính bằng nội suy điểm giữa
q3 = np.percentile(age_data, 75, method='midpoint')

print(f"Mean: {mean_val:.2f}, Median: {median_val}, Q1: {q1}, Q3: {q3}")

# 2. Rời rạc hóa k=4
# Chia đều chiều rộng (Equal-width)
eq_width_bins = np.linspace(min(age_data), max(age_data), 5)
print("Ranh giới Equal-width:", eq_width_bins)

# Chia đều tần số (Equal-frequency)
eq_freq_bins = pd.qcut(age_data, q=4, retbins=True)[1]
print("Ranh giới Equal-frequency:", eq_freq_bins)