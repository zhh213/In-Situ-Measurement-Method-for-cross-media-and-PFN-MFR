import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def simple_lapsed_time(text, lapsed):  # 记录代码执行时间
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def concat_data(X, y):   # 将X中特征和Y中第一列转换成dataframe格式，特征和目标值组合，头衔是target
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:, 0].tolist(), columns=['target'])], axis=1)


def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise ValueError('Shape of data not same as that of nan mask!')

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d


def data_prep_custom(data_path, seed, task, datasplit=[.45, .15, .4]):  # [.65 .15 .2]
    np.random.seed(seed)

    # 加载自定义数据集
    data = pd.read_csv('D:\\edge_download\\TABLE_net\\saint_zh\\zhh_r.csv')
    # data = pd.read_excel('data_path')
    # 假设最后一列是目标变量
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 处理分类特征和连续特征
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cont_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    cat_idxs = [X.columns.get_loc(col) for col in categorical_columns]
    con_idxs = [X.columns.get_loc(col) for col in cont_columns]

    # 填充缺失值
    # for col in categorical_columns:
    #     X[col] = X[col].fillna("MissingValue")
    #    l_enc = LabelEncoder()
    #    X[col] = l_enc.fit_transform(X[col].values)

    #for col in cont_columns:
    #    X[col] = X[col].fillna(X[col].mean())

    # 将目标变量编码为数值
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
    else:
        y = y.values

    # 添加一个临时列用于数据分割
    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

    # 获取分割后的索引
    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    # 移除临时列
    X = X.drop(columns=['Set'])

    # 创建缺失值掩码
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    # 分割数据
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    # 标准化连续特征
    train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(
        X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return [], cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # 分类列
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # 数值列
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # 分类列掩码
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # 数值列掩码
        if task == 'clf':
            self.y = Y['data']  # .astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros((len(self.y), 1), dtype=int)  # 确保 self.cls 的大小与数据集一致
        self.cls_mask = np.ones((len(self.y), 1), dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 添加边界检查
        if idx < 0 or idx >= len(self.cls):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.cls)}")

        # 访问 self.cls
        cls = self.cls[idx].flatten()
        cls_mask = self.cls_mask[idx].flatten()  # 确保 cls_mask 是一维数组
        return np.concatenate((cls, self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate(
            (cls_mask, self.X1_mask[idx])), self.X2_mask[idx]
