import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo dữ liệu
data = {
    'ID': [1,2,3,4,5,6,7,8,9,10],
    'Tuổi': [25, 30, 22, 40, 35, np.nan, 29, 50, 27, 31],
    'Lương': [8000, np.nan, 5000, 15000, -2000, 7000, 8500, 30000, 8200, 7800],
    'Điểm': [7.5, 8.0, 6.5, 9.0, 7.0, 6.0, 8.5, 9.5, np.nan, 7.8]
}
df = pd.DataFrame(data)

# 1. Tính Thống kê mô tả
stats = pd.DataFrame({
    'Mean': df[['Tuổi', 'Lương', 'Điểm']].mean(),
    'Median': df[['Tuổi', 'Lương', 'Điểm']].median(),
    'Mode': df[['Tuổi', 'Lương', 'Điểm']].mode().iloc[0]
})
print(stats)

# 2. Vẽ Histogram và Boxplot (Câu 1.3)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Tuổi'].dropna(), bins=6, kde=True)
plt.title('Histogram: Phân phối Tuổi')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Lương'].dropna())
plt.title('Boxplot: Phân phối Lương')
plt.show()

# 3. Vẽ Scatter Plot (Câu 1.4)
df_scatter = df.dropna(subset=['Tuổi', 'Lương'])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scatter, x='Tuổi', y='Lương', s=100)
plt.title('Scatter Plot: Tuổi vs Lương')
plt.show()

# 4. Tính Z-score (Câu 1.9)
mean_luong = df['Lương'].mean()
std_luong = df['Lương'].std()
df['Lương_Zscore'] = (df['Lương'] - mean_luong) / std_luong