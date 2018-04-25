import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

## 1.读取数据
melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
#print(melbourne_data.describe())

y = melbourne_data.Price
## 2.选择数据
melbourne_predictors = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
X = melbourne_data[melbourne_predictors]
# 用零填充空值
X = X.fillna(0)

## 3.构建模型  决策树
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X,y)

# 评估
#predicted_home_prices = melbourne_model.predict(X)
#print('mse',mean_absolute_error(y,predicted_home_prices))
# 单样本评估，在训练样本中得分很高。需要划分数据集

## 4.划分测试集和训练集 进行模型评估
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X,train_y)

# 评估结果
val_predictions = melbourne_model.predict(val_X)
print('mse',mean_absolute_error(val_y,val_predictions))

def get_mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train,targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val,preds_val)
    return mae

for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# 随机森林模型
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


## 5. 提交结果
# Read the test data
# test = pd.read_csv('../input/test.csv')
# # Treat the test data in the same way as training data. In this case, pull same columns.
# test_X = test[predictor_cols]
# # Use the model to make predictions
# predicted_prices = my_model.predict(test_X)
# # We will look at the predicted prices to ensure we have something sensible.
# print(predicted_prices)
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# # you could use any filename. We choose submission here
# my_submission.to_csv('./data/submission.csv', index=False)