# 处理缺失值
import pandas as pd 

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# 1.简单粗暴的处理方式 直接丢掉有缺失值的列
new_data = melbourne_data.copy()
#new_data = new_data.dropna(axis=1)
#print(new_data)
#print(melbourne_data)

# 2.均值填充  数值类型有用
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
#data_with_imputed_values = my_imputer.fit_transform(new_data)

# 3.标记缺失值，并用均值填充
cols_with_missing = (col for col in new_data.columns if new_data[col].isnull().any())
#print(cols_with_missing)
for col in cols_with_missing:
    new_data[col+'_was_missing'] = new_data[col].isnull()

#print(new_data)

print('Three method')


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 得到数据集的得分
def score_dataset(X,y):
    train_X,test_X,train_y,test_y = train_test_split(X,y,random_state = 0)
    forest_model = RandomForestRegressor()
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(test_X)
    return mean_absolute_error(test_y, melb_preds)

# 选择数据
melbourne_predictors = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
origin_data = melbourne_data[melbourne_predictors]
y = melbourne_data.Price

# 缺失值处理
# 1.
new_data = origin_data.copy()
new_data = new_data.dropna(axis=1)
X = new_data
print('first method:',score_dataset(X,y))

# 2.
new_data = origin_data.copy()
my_imputer = Imputer()
X = my_imputer.fit_transform(new_data)
print('second method:',score_dataset(X,y))

# 3.
new_data = origin_data.copy()
cols_with_missing = (col for col in new_data.columns if new_data[col].isnull().any())
#print(cols_with_missing)
for col in cols_with_missing:
    new_data[col+'_was_missing'] = new_data[col].isnull()

my_imputer = Imputer()
X = my_imputer.fit_transform(new_data)
print('third method:',score_dataset(X,y))