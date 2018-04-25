# 处理类型变量
import pandas as pd 

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

#print(melbourne_data)
# 选择数据
#melbourne_predictors = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
origin_data = melbourne_data.copy()#[melbourne_predictors]
#print(origin_data.dtypes)
#one_hot_encoded_training_predictors = pd.get_dummies(origin_data)
#print(one_hot_encoded_training_predictors)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()


predictors_without_categoricals = origin_data.select_dtypes(exclude=['object'])
print(predictors_without_categoricals)

#mae_without_categoricals = get_mae(predictors_without_categoricals, target)
#mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

#print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
#print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))