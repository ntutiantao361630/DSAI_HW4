import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="lightgbm")
import lightgbm as lgbm
import joblib

def get_final_data():
    matrix = pd.read_pickle("../output/checkpoint_final.pkl")
    matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0,20)
    keep_from_month = 2  # The first couple of months are dropped because of distortions to their features (e.g. wrong item age)
    test_month = 33
    dropcols = [
        "shop_id",
        "item_id",
        "new_item",
    ]  # The features are dropped to reduce overfitting

    valid = matrix.drop(columns=dropcols).loc[matrix.date_block_num == test_month, :]
    train = matrix.drop(columns=dropcols).loc[matrix.date_block_num < test_month, :]
    train = train[train.date_block_num >= keep_from_month]
    X_train = train.drop(columns="item_cnt_month")
    y_train = train.item_cnt_month
    X_valid = valid.drop(columns="item_cnt_month")
    y_valid = valid.item_cnt_month
    print(matrix)
    del matrix
    return X_train, y_train, X_valid, y_valid

def fit_booster(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    params=None,
    test_run=False,
    categoricals=[],
    dropcols=[],
    early_stopping=True,
):
    if params is None:
        params = {"learning_rate": 0.1, "subsample_for_bin": 300000, "n_estimators": 50}

    early_stopping_rounds = None
    if early_stopping == True:
        early_stopping_rounds = 100

    if test_run:
        eval_set = [(X_train, y_train)]
    else:
        eval_set = [(X_train, y_train), (X_test, y_test)]

    booster = lgbm.LGBMRegressor(**params)

    categoricals = [c for c in categoricals if c in X_train.columns]

    booster.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric=["rmse"],
        verbose=100,
        categorical_feature=categoricals,
        early_stopping_rounds=early_stopping_rounds,
    )
    return booster

def train_model():
    X_train, y_train, X_valid, y_valid = get_final_data()
    params = {
        "num_leaves": 966,
        "cat_smooth": 45.01680827234465,
        "min_child_samples": 27,
        "min_child_weight": 0.021144950289224463,
        "max_bin": 214,
        "learning_rate": 0.01,
        "subsample_for_bin": 300000,
        "min_data_in_bin": 7,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "subsample_freq": 5,
        "n_estimators": 8000,
    }
    categoricals = [
        "item_category_id",
        "month",
    ]
    lgbooster = fit_booster(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=params,
        test_run=False,
        categoricals=categoricals,
    )
    _ = joblib.dump(lgbooster, "../output/trained_lgbooster.pkl")
