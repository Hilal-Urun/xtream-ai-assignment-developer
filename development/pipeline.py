import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
import optuna


def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.x * df.y * df.z != 0) & (df.price > 0)]
    df_processed = df.drop(columns=['depth', 'table', 'y', 'z'])
    df_dummy = pd.get_dummies(df_processed, columns=['cut', 'color', 'clarity'], drop_first=True)
    return df_dummy


def train_model(model, x_train: pd.DataFrame, y_train: pd.Series):
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(x_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    return {'R2 Score': r2, 'MAE': mae}


def save_model(model, model_dir: str, model_name: str, timestamp: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, f'{model_name}_{timestamp}.pkl'))


def save_metrics(metrics: dict, metrics_file: str, model_name: str, timestamp: str):
    metrics_df = pd.DataFrame([metrics], index=[f'{model_name}_{timestamp}'])
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False)
    else:
        metrics_df.to_csv(metrics_file, mode='w')


def plot_gof(y_true: pd.Series, y_pred: pd.Series):
    plt.plot(y_true, y_pred, '.')
    plt.plot(y_true, y_true, linewidth=3, c='black')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()


def objective(trial: optuna.trial.Trial, x_train, y_train) -> float:
    param = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True
    }
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(**param)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    mae = mean_absolute_error(y_val, preds)
    return mae


def linearl_model_dataprep(df: pd.DataFrame) -> tuple:
    diamonds_dummy = preprocess_data(df)
    x = diamonds_dummy.drop(columns='price')
    y = diamonds_dummy.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def main():
    # url = "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"
    model_dir = 'models'
    metrics_file = 'model_metrics.csv'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models = {
        'LinearRegression': LinearRegression(),
    }
    # diamonds = load_data(url)
    diamonds = pd.read_csv("data/diamonds.csv")

    for model_name, model in models.items():
        x_train, x_test, y_train, y_test = linearl_model_dataprep(diamonds)
        trained_model = train_model(model, x_train, y_train)
        metrics = evaluate_model(trained_model, x_test, y_test)
        print(f"{model_name} trained at {timestamp}: R2 Score - {metrics['R2 Score']}, MAE - {metrics['MAE']}$")
        save_model(trained_model, model_dir, model_name, timestamp)
        save_metrics(metrics, metrics_file, model_name, timestamp)

        # plot_gof(y_test, trained_model.predict(x_test))

    # XGBoost with hyperparameter optimization
    diamonds_processed_xgb = diamonds.copy()
    diamonds_processed_xgb['cut'] = pd.Categorical(diamonds_processed_xgb['cut'],
                                                   categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                                   ordered=True)
    diamonds_processed_xgb['color'] = pd.Categorical(diamonds_processed_xgb['color'],
                                                     categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    diamonds_processed_xgb['clarity'] = pd.Categorical(diamonds_processed_xgb['clarity'],
                                                       categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2',
                                                                   'I1'], ordered=True)
    # diamonds_processed_xgb.info()
    x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(diamonds_processed_xgb.drop(columns='price'),
                                                                        diamonds_processed_xgb['price'], test_size=0.2,
                                                                        random_state=42)

    study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(lambda trial: objective(trial, x_train_xbg, y_train_xbg), n_trials=100)
    print("Best hyperparameters: ", study.best_params)
    xgb_opt = xgb.XGBRegressor(**study.best_params, enable_categorical=True, random_state=42)
    xgb_opt.fit(x_train_xbg, y_train_xbg)
    xgb_opt_pred = xgb_opt.predict(x_test_xbg)

    xgb_opt_metrics = {'R2 Score': r2_score(y_test_xbg, xgb_opt_pred),
                       'MAE': mean_absolute_error(y_test_xbg, xgb_opt_pred)}
    print(
        f"XGBoost optimized trained at {timestamp}: R2 Score - {xgb_opt_metrics['R2 Score']}, MAE - {xgb_opt_metrics['MAE']}$")

    save_model(xgb_opt, model_dir, 'XGBoost_Optuna', timestamp)
    save_metrics(xgb_opt_metrics, metrics_file, 'XGBoost_Optuna', timestamp)

    # plot_gof(y_test, xgb_opt_pred)


if __name__ == '__main__':
    main()
