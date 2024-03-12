from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import xgboost as xgb

train = pd.read_csv('/Users/violette/Downloads/train_df.csv')
test = pd.read_csv('/Users/violette/Downloads/test_df.csv')

drop_cols = []
for col in train.columns:
    if len(train[col].value_counts()) == 1:
        drop_cols.append(col)

train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)

train['group_count'] = train.groupby('search_id').cumcount('target')
group_counts = train.groupby('search_id').last()['group_count'].to_numpy(dtype=np.int32)

y = train['target']
x = train.drop(['search_id', 'target'], axis=1)

params = {
'objective': 'rank:pairwise',
'learning_rate': 0.1,
'max_depth': 6,
'n_estimators': 100
}

model = xgb.sklearn.XGBRanker(**params)
model.fit(x, y, group=group_counts)
