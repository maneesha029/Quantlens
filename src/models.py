from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_models(X_train, y_train):
    models = {}

    models["linear"] = LinearRegression().fit(X_train, y_train)
    models["rf"] = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    ).fit(X_train, y_train)
    models["xgboost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ).fit(X_train, y_train)

    print("All models trained successfully.")
    return models

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        print(f"{name.upper():10s} | R²: {r2:.4f} | RMSE: {rmse:.5f}")
