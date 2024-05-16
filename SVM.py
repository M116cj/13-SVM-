import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
data = pd.read_csv('coin_BinanceCoin.csv')
print(data.head(5))
selected_data = data[['High', 'Low','Open', 'Close','Volume', 'Marketcap']]
covariance_matrix = normalized_data.cov()
print(covariance_matrix)   #共變異數矩陣
sns.heatmap(covariance_matrix, annot=True, cmap="YlGnBu")
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_data)
normalized_data = pd.DataFrame(normalized_data, columns=selected_data.columns)
print(normalized_data)
time_series = data['Close']
ma_5 = time_series.rolling(window=5).mean()
ma_5 = time_series.rolling(window=5).mean()
ma_10 = time_series.rolling(window=10).mean()
ma_20 = time_series.rolling(window=20).mean()
ma_60 = time_series.rolling(window=60).mean()
ma_120 = time_series.rolling(window=120).mean()
ma_240 = time_series.rolling(window=240).mean()
moving_averages = pd.DataFrame({
    'MA(5)': ma_5,
    'MA(10)': ma_10,
    'MA(20)': ma_20,
    'MA(60)': ma_60,
    'MA(120)': ma_120,
    'MA(240)': ma_240
})
trimmed_moving_averages = moving_averages[300:]
trimmed_moving_averages
data = data.rename(columns={'30天前的报酬': '30day_return'})#設定y軸
data
selected_vars_table1 = data[['30day_return','Volume']]
selected_vars_table2= moving_averages [['MA(5)','MA(10)','MA(20)','MA(60)','MA(120)','MA(240)']]
merged_table = pd.concat([selected_vars_table1, selected_vars_table2], axis=1)
merged_table 
trimmed_merged_table  = merged_table [300:-300]   #因為240天的ma抓太少資料會不夠用
trimmed_merged_table 
selected_columns = ['30day_return','Volume','MA(5)','MA(10)','MA(20)','MA(60)','MA(120)','MA(240)']
trimmed_merged_table[selected_columns] = scaler.fit_transform(trimmed_merged_table[selected_columns])
trimmed_merged_table

##normalized_data_1 = scaler.fit_transform(trimmed_merged_table )
##normalized_data = pd.DataFrame(normalized_data_1, columns=trimmed_merged_table.columns)

covariance_matrix = trimmed_merged_table.cov()#尋找跟y軸有相關性的變數
covariance_matrix
sns.heatmap(covariance_matrix, annot=True, cmap="YlGnBu")
data['30天後報酬'] = data['Close'].pct_change(periods=-30)
print(data.head(50))
data['30天前的报酬'] = data['Close'].pct_change(periods=30).shift(-30)
data
trimmed_merged_table['new_y'] = trimmed_merged_table['30day_return'].apply(lambda x: 1 if x > 0 else 0)
trimmed_merged_table
model = svm.SVC(kernel='linear')
X = trimmed_merged_table[['Volume', 'MA(120)']]
y = trimmed_merged_table['new_y']
##'30day_return','Volume','MA(5)','MA(10)','MA(20)','MA(60)','MA(120)','MA(240)'



# 擬和模型
model.fit(X,y)

# 獲取索引
support_vector_indices = model.support_

# 獲取座標
support_vectors = model.support_vectors_

# 獲取標籤
support_vector_labels = y[support_vector_indices]

# 繪製散點圖
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c=support_vector_labels, cmap=plt.cm.Paired, marker='x', label='Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
model = svm.SVC(kernel='linear')
print("X shape:", X.shape)
print("support_vector_indices:", support_vector_indices)
trimmed_merged_table
X
y

plt.scatter(trimmed_merged_table['Volume'], trimmed_merged_table['MA(120)'], c=y, cmap=plt.cm.Paired, label='Data')
plt.scatter(support_vectors['Volume'], support_vectors['MA(120)'], c=support_vector_labels, cmap=plt.cm.Paired, marker='x', label='Support Vectors')
plt.xlabel('Volume')
plt.ylabel('MA(120)')

##plt.xlim(-1,1)

plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm



# SVM 模型
model = svm.SVC(kernel='linear')

# 擬合模型
model.fit(trimmed_merged_table[['Volume', 'MA(120)']], y)

# 獲取索引
support_vector_indices = model.support_

# 獲取座標
support_vectors = trimmed_merged_table[['Volume', 'MA(120)']].iloc[support_vector_indices]

# 獲取標籤
support_vector_labels = y[support_vector_indices]

# 繪製散點圖
plt.scatter(trimmed_merged_table['Volume'], trimmed_merged_table['MA(120)'], c=y, cmap=plt.cm.Paired, label='Data')
plt.scatter(support_vectors['Volume'], support_vectors['MA(120)'], c=support_vector_labels, cmap=plt.cm.Paired, marker='x', label='Support Vectors')

# 獲取參數
w = model.coef_[0]
b = model.intercept_[0]

# 生成x軸上的點
x = np.linspace(trimmed_merged_table['Volume'].min(), trimmed_merged_table['Volume'].max(), 100)
# 根據參數，計算 x 軸上對應的 y 軸值
y = -(w[0] / w[1]) * x - (b / w[1])

# 繪製邊界線
plt.plot(x, y, color='black', linestyle='--', label='Decision Boundary')

plt.xlabel('Volume')
plt.ylabel('MA(120)')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm



# 創建模型
model = svm.SVC(kernel='linear')

# 擬合模型
model.fit(trimmed_merged_table[['Volume', 'MA(120)']], y)

# 獲取索引
support_vector_indices = model.support_

# 獲取座標
support_vectors = trimmed_merged_table[['Volume', 'MA(120)']].iloc[support_vector_indices]

# 獲取標籤
support_vector_labels = y[support_vector_indices]

# 繪製散點圖
plt.scatter(trimmed_merged_table['Volume'], trimmed_merged_table['MA(120)'], c=y, cmap=plt.cm.Paired, label='Data')
plt.scatter(support_vectors['Volume'], support_vectors['MA(120)'], c=support_vector_labels, cmap=plt.cm.Paired, marker='x', label='Support Vectors')

# 獲取模型參數
w = model.coef_[0]
b = model.intercept_[0]

# 生成 X 軸上的點
x_min, x_max = trimmed_merged_table['Volume'].min(), trimmed_merged_table['Volume'].max()
y_min, y_max = trimmed_merged_table['MA(120)'].min(), trimmed_merged_table['MA(120)'].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製邊界線
plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', label='Decision Boundary')

plt.xlabel('Volume')
plt.ylabel('MA(120)')

plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

plt.xlabel('Volume')
plt.ylabel('MA(120)')

plt.legend()
plt.show()
print(trimmed_merged_table[['Volume', 'MA(120)']].shape)
print(y.shape)
from sklearn import svm
import matplotlib.pyplot as plt

X = trimmed_merged_table[['Volume', 'MA(120)']]

Y = trimmed_merged_table['new_y']

model = svm.SVC(kernel='linear')

model.fit(X, Y)

support_vector_indices = model.support_

support_vectors = X.iloc[support_vector_indices]

support_vector_labels = Y.iloc[support_vector_indices]

plt.scatter(X['Volume'], X['MA(120)'], c=Y, cmap=plt.cm.Paired, label='Data')
plt.scatter(support_vectors['Volume'], support_vectors['MA(120)'], c=support_vector_labels, cmap=plt.cm.Paired, marker='x', label='Support Vectors')

w = model.coef_[0]
b = model.intercept_[0]

x_min, x_max = X['Volume'].min(), X['Volume'].max()#淡藍色是真實漲幅，咖啡色是模型預測的漲幅
y_min, y_max = X['MA(120)'].min(), X['MA(120)'].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', label='Decision Boundary')

plt.xlabel('Volume')
plt.ylabel('MA(120)')

plt.legend()
plt.show()
