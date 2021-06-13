import pandas as pd
import joblib
def make_sub():
    dropcols = [
        "shop_id",
        "item_id",
        "new_item",
    ]
    matrix = pd.read_pickle("../output/checkpoint_final.pkl")
    matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0,20)
    keep_from_month = 2
    test_month = 34
    test = matrix.loc[matrix.date_block_num==test_month, :]
    X_test = test.drop(columns="item_cnt_month")
    y_test = test.item_cnt_month
    del(matrix)
    lgbooster = joblib.load("../output/trained_lgbooster.pkl")
    X_test["item_cnt_month"] = lgbooster.predict(X_test.drop(columns=dropcols)).clip(0, 20)
    # Merge the predictions with the provided template
    test_orig = pd.read_csv("../data/test.csv")
    test = test_orig.merge(
        X_test[["shop_id", "item_id", "item_cnt_month"]],
        on=["shop_id", "item_id"],
        how="inner",
        copy=True,
    )
    # Verify that the indices of the submission match the original
    assert test_orig.equals(test[["ID", "shop_id", "item_id"]])
    test[["ID", "item_cnt_month"]].to_csv("../output/submission.csv", index=False)
    print("Finished!")