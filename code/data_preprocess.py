import gc
import itertools
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns

def call_api():
    print(np.array([1,2,3]))
    print(pd.DataFrame({1:['a', 'b']}))

class Data_Preprocess():
    def __init__(self):
        self.train = None
        self.test = None
        self.items = None
        self.shops = None

    def __call__(self, *args, **kwargs):
        self.load_data()
        self.data_cleansing()
        matrix = self.create_testlike_train()
        del (self.test)
        gc.collect()
        matrix = reduce_mem_usage(matrix, silent=False)
        oldcols = matrix.columns
        return matrix, oldcols, self.train, self.items, self.shops

    def load_data(self):
        self.items = pd.read_csv("../data/items.csv")
        self.shops = pd.read_csv("../data/shops.csv")
        self.train = pd.read_csv("../data/sales_train.csv")
        self.test = pd.read_csv("../data/test.csv")
        self.train["date"] = pd.to_datetime(self.train["date"], format="%d.%m.%Y")

    def data_cleansing(self):
        # Merge some duplicate shops
        self.train["shop_id"] = self.train["shop_id"].replace({0: 57, 1: 58, 11: 10, 40: 39})
        # Keep only shops that are in the test set
        self.train = self.train.loc[self.train.shop_id.isin(self.test["shop_id"].unique()), :]
        # Drop training items with extreme or negative prices or sales counts
        self.train = self.train[(self.train["item_price"] > 0) & (self.train["item_price"] < 50000)]
        self.train = self.train[(self.train["item_cnt_day"] > 0) & (self.train["item_cnt_day"] < 1000)]

    def create_testlike_train(self):
        indexlist = []
        for i in self.train.date_block_num.unique():
            x = itertools.product(
                [i],
                self.train.loc[self.train.date_block_num == i].shop_id.unique(),
                self.train.loc[self.train.date_block_num == i].item_id.unique(),
            )
            indexlist.append(np.array(list(x)))
        df = pd.DataFrame(
            data=np.concatenate(indexlist, axis=0),
            columns=["date_block_num", "shop_id", "item_id"],
        )

        # Add revenue column to self.train
        self.train["item_revenue_day"] = self.train["item_price"] * self.train["item_cnt_day"]
        # Aggregate item_id / shop_id item_cnts and revenue at the month level
        self.train_grouped = self.train.groupby(["date_block_num", "shop_id", "item_id"]).agg(
            item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum"),
            item_revenue_month=pd.NamedAgg(column="item_revenue_day", aggfunc="sum"),
        )

        # Merge the grouped data with the index
        df = df.merge(
            self.train_grouped, how="left", on=["date_block_num", "shop_id", "item_id"],
        )

        if self.test is not None:
            self.test["date_block_num"] = 34
            self.test["date_block_num"] = self.test["date_block_num"].astype(np.int8)
            self.test["shop_id"] = self.test.shop_id.astype(np.int8)
            self.test["item_id"] = self.test.item_id.astype(np.int16)
            self.test = self.test.drop(columns="ID")
            df = pd.concat([df, self.test[["date_block_num", "shop_id", "item_id"]]])

        # Fill empty item_cnt entries with 0
        df.item_cnt_month = df.item_cnt_month.fillna(0)
        df.item_revenue_month = df.item_revenue_month.fillna(0)
        return df
