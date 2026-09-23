# -*- coding: utf-8 -*-
"""
House Prices: Advanced Regression Techniques
Trích xuất toàn bộ mã nguồn từ tài liệu PDF: "House Prices: Advanced Regression Techniques - Rishabh Nimje"
Đã tối ưu và tương thích với Keras 3 / Pandas 2.x / Kaggle Environment
"""

# ==========================================
# 1. IMPORT THƯ VIỆN CẦN THIẾT
# ==========================================
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Machine Learning & Tuning
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier  # Bài gốc dùng Classifier (trong thực tế hồi quy nên dùng DecisionTreeRegressor)

# Thư viện Deep Learning (Chuẩn Keras 3)
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import ops

import kagglehub

# Tự động tải hoặc định vị dữ liệu
try:
    path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
    print("Path to competition files:", path)
except Exception as e:
    path = '/kaggle/input/house-prices-advanced-regression-techniques'
    print("Using default Kaggle path:", path)


# ==========================================
# 2. TẢI DỮ LIỆU (LOAD DATA)
# ==========================================
print("\n=== 1. TẢI DỮ LIỆU ===")
# Tìm file train và test linh hoạt
train_path = os.path.join(path, 'train.csv') if os.path.exists(os.path.join(path, 'train.csv')) else 'train.csv'
test_path = os.path.join(path, 'test.csv') if os.path.exists(os.path.join(path, 'test.csv')) else 'test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)


# ==========================================
# 3. KIỂM TRA GIÁ TRỊ THIẾU (NULL VALUES)
# ==========================================
print("\n=== 2. KIỂM TRA DỮ LIỆU KHUYẾT (NULL VALUES) ===")
print("Train NULL values count:", train.isnull().sum().sum())
print("Test NULL values count:", test.isnull().sum().sum())


# ==========================================
# 4. XỬ LÝ DỮ LIỆU KHUYẾT (HANDLING NULL DATA)
# ==========================================
print("\n=== 3. XỬ LÝ DỮ LIỆU KHUYẾT ===")
# Xử lý cho tập Train
cat_col_train = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'GarageQual', 'GarageCond'
]
ncat_col_train = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

for i in cat_col_train:
    train[i] = train[i].fillna(train[i].mode()[0])

for j in ncat_col_train:
    train[j] = train[j].fillna(train[j].mean())

# Xử lý cho tập Test
cat_col_test = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'GarageQual', 'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st',
    'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
]
ncat_col_test = [
    'LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars',
    'GarageArea'
]

for i in cat_col_test:
    test[i] = test[i].fillna(test[i].mode()[0])

for j in ncat_col_test:
    test[j] = test[j].fillna(test[j].mean())

# Xóa các cột có tỷ lệ rỗng > 70% và cột 'Id'
to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
for k_col in to_drop:
    train.drop([k_col], axis=1, inplace=True, errors='ignore')
    test.drop([k_col], axis=1, inplace=True, errors='ignore')

print("Sau khi xử lý NULL:")
print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)


# ==========================================
# 5. MÃ HÓA ONE-HOT (ONE HOT ENCODING)
# ==========================================
print("\n=== 4. TIẾN HÀNH ONE-HOT ENCODING ===")
final_df = pd.concat([train, test], axis=0)

all_cat_col = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'SaleType', 'SaleCondition'
]

def cat_onehot_encoding(multicol):
    global final_df
    df_final = final_df
    i = 0
    for fields in multicol:
        # Sử dụng dtype=float để tránh sinh ra kiểu bool (nguyên nhân gây lỗi object dtype)
        df1 = pd.get_dummies(final_df[fields], drop_first=True, dtype=float)
        final_df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
        
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final

final_df = cat_onehot_encoding(all_cat_col)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
print("Shape sau khi loại bỏ cột trùng lặp:", final_df.shape)


# ==========================================
# 6. TÁCH LẠI TẬP TRAIN VÀ TEST
# ==========================================
print("\n=== 5. CHIA LẠI DỮ LIỆU HUẤN LUYỆN VÀ KIỂM THỬ ===")
df_train = final_df.iloc[:1460, :]
df_test = final_df.iloc[1460:, :].copy()

# Xóa cột SalePrice khỏi tập test
df_test = df_test.drop(['SalePrice'], axis=1)

# Tách features (x_train) và target (y_train)
x_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']

print("x_train Shape: ", x_train.shape)
print("df_test Shape: ", df_test.shape)


# ==========================================
# HÀM HỖ TRỢ XUẤT SUBMISSION FILE
# ==========================================
def export_submission(predictions, filename):
    sub_path = os.path.join(path, 'sample_submission.csv') if os.path.exists(os.path.join(path, 'sample_submission.csv')) else 'sample_submission.csv'
    if os.path.exists(sub_path):
        sub_df = pd.read_csv(sub_path)
        sub_df['SalePrice'] = predictions
        sub_df.to_csv(filename, index=False)
        print(f"-> Đã xuất file nộp: {filename}")
    else:
        # Nếu không có file mẫu thì tự tạo dựa trên ID test (1461 -> 2919)
        test_ids = list(range(1461, 1461 + len(predictions)))
        sub_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
        sub_df.to_csv(filename, index=False)
        print(f"-> Đã tạo mới file nộp: {filename}")


# ==========================================
# 7. MÔ HÌNH 1: XGBOOST REGRESSOR
# ==========================================
print("\n=== 6. HUẤN LUYỆN MÔ HÌNH XGBOOST ===")
xgb_best = xgboost.XGBRegressor(
    base_score=0.25,
    booster='gbtree',
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=1,
    n_estimators=900,
    objective='reg:squarederror',
    random_state=0
)

xgb_best.fit(x_train, y_train)

# Lưu model
with open("xgb_model.pkl", "wb") as f_out:
    pickle.dump(xgb_best, f_out)
print("Đã lưu: xgb_model.pkl")

# Dự đoán và xuất file
pred_xgb = xgb_best.predict(df_test)
export_submission(pred_xgb, 'sample_sub_xgb.csv')


# ==========================================
# 8. MÔ HÌNH 2: DECISION TREE
# ==========================================
print("\n=== 7. HUẤN LUYỆN MÔ HÌNH DECISION TREE ===")
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

pred_dt = dt_model.predict(df_test)
export_submission(pred_dt, 'sample_sub_dt.csv')


# ==========================================
# 9. MÔ HÌNH 3: MẠNG NƠ-RON NHÂN TẠO (ANN)
# ==========================================
print("\n=== 8. HUẤN LUYỆN MẠNG NƠ-RON (ANN) ===")

from keras.models import Sequential
from keras.layers import Dense, Input
from keras import ops
import numpy as np

# Hàm loss tùy chỉnh chuẩn Keras 3
def root_mean_squared_error(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)) + 1e-7)

# Ép kiểu dữ liệu sang NumPy float32
x_train_ann = np.asarray(x_train, dtype=np.float32)
y_train_ann = np.asarray(y_train, dtype=np.float32)
df_test_ann = np.asarray(df_test, dtype=np.float32)

# Xây dựng mô hình với layer Input chuẩn Keras 3
nn_model = Sequential([
    Input(shape=(x_train_ann.shape[1],)),
    Dense(50, kernel_initializer='he_uniform', activation='relu'),
    Dense(25, kernel_initializer='he_uniform', activation='relu'),
    Dense(50, kernel_initializer='he_uniform', activation='relu'),
    Dense(1, kernel_initializer='he_uniform')
])

nn_model.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Huấn luyện mô hình
nn_model.fit(
    x_train_ann,
    y_train_ann,
    validation_split=0.25,
    batch_size=10,
    epochs=1000,
    verbose=1
)

# Lưu mô hình (Keras 3)
nn_model.save('nn_model.keras')
print("Đã lưu mô hình: nn_model.keras")

# Dự đoán và xuất file nộp
pred_nn = nn_model.predict(df_test_ann).flatten()
export_submission(pred_nn, 'sample_sub_nn.csv')

print("\n=== HOÀN TẤT TOÀN BỘ QUY TRÌNH THÀNH CÔNG ===")