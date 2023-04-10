import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设历史数据是一个二维列表，其中每个子列表包含3个骰子的结果
historical_data = [[3, 1, 1], [6, 5, 2], [4, 4, 4], [6, 3, 1], [6, 2, 1], [5, 3, 2], [5, 4, 3], [6, 4, 3], [6, 4, 2], [5, 3, 2],
           [5, 4, 1], [5, 3, 1], [3, 2, 1], [6, 5, 4], [4, 4, 2], [6, 5, 4], [3, 3, 1], [5, 2, 1], [6, 6, 3], [6, 4, 3],
           [4, 3, 3], [6, 4, 4], [2, 2, 1], [6, 5, 1], [4, 2, 2], [5, 2, 1], [2, 2, 1], [6, 3, 1], [4, 3, 1], [6, 2, 1],
           [4, 4, 1], [5, 1, 1], [6, 5, 3], [5, 2, 1], [6, 6, 4], [4, 3, 3], [3, 2, 2], [5, 5, 2], [6, 6, 2], [6, 3, 2],
           [3, 2, 2], [6, 5, 2], [4, 3, 2], [2, 1, 1], [5, 3, 2], [5, 3, 2], [3, 2, 2], [2, 2, 1], [5, 3, 2], [6, 5, 4],
           [6, 4, 4], [6, 3, 1], [4, 3, 2], [6, 4, 2], [6, 4, 4], [4, 4, 3], [5, 5, 5], [4, 3, 1], [2, 2, 2], [5, 4, 3],
           [6, 1, 1], [5, 2, 2], [2, 1, 1], [4, 2, 2], [6, 5, 1], [5, 4, 1], [6, 6, 1], [3, 5, 1], [6, 4, 2], [6, 4, 3],
           [6, 4, 2], [4, 3, 1], [6, 3, 1], [5, 4, 1], [5, 2, 1], [4, 2, 2], [6, 6, 3]]

data = pd.DataFrame(historical_data[:-1], columns=["feature1", "feature2", "feature3"])
data["outcome"] = data.apply(lambda row: sum(row), axis=1)
X = data.drop("outcome", axis=1)
y = data["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 构建梯度提升机模型
# gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm_model.fit(X, y)

# 预测测试集
y_pred = gbm_model.predict([historical_data[-1]])
print(y_pred)


# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: {:.4f}".format(mse))

# 基于历史数据计算平均赔率
average_odds = data["outcome"].mean()

# 使用均方误差预测胜率
predicted_win_probability = 1 - mse

# 应用Kelly准则
def kelly_criterion(odds, probability):
    return ((odds * probability) - 1) / (odds - 1)

bet_fraction = kelly_criterion(average_odds, predicted_win_probability)
print("Bet fraction:", bet_fraction)