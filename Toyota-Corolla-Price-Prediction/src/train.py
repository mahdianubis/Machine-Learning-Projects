import pandas as pd
from evaluate import evaluate_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Machine-Learning-Projects/Toyota-Corolla-Price-Prediction/data/processed/ToyotaCorolla_processed.csv")

x = df.drop(["Model", "Price"], axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_split=6, random_state=42)

model.fit(x_train, y_train)

# parameters = [{"n_estimators" : [25, 50, 100, 150], "max_depth" : [2, 4, 6, 8, 10, 12, 15], "min_samples_split" : [2, 4, 6, 8, 10, 12, 15]}]
# grid_search = GridSearchCV(estimator=RandomForestRegressor(), scoring="r2", param_grid=parameters, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)

# best_model = grid_search.best_estimator_
# print(best_model)

evaluate_model("RandomForestRegressor", model, x_test, y_test)