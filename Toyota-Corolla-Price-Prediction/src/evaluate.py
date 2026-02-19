from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model_name, model, x_test, y_test):
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    with open("Machine-Learning-Projects/Toyota-Corolla-Price-Prediction/reports/results.txt", "a") as f:
        f.write(f"Model name : {model_name}\n")
        f.write(f"R2 : {r2 * 100:.3f}\n")
        f.write(f"MAE : {mae:.2f}\n")
        f.write(f"MSE : {mse:.2f}\n")