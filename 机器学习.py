# 导入所需的包
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

# 打印数据集中缺失值的数量
print("Train data missing values:")
print(train.isnull().sum())
print("\nTest data missing values:")
print(test.isnull().sum())

# 合并数据集，方便一起处理
data = pd.concat([train, test], ignore_index=True)

# 数据探索：计算并展示每个字段的唯一值数量
unique_values = data.apply(pd.Series.nunique)
print(unique_values.sort_values(ascending=False))

# 处理"property_damage"和"police_report_available"中的"?"
for col in ['property_damage', 'police_report_available']:
    data[col] = data[col].replace({'NO': 0, 'YES': 1, '?': None})

# 将日期字段转换为距离最早日期的天数
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
data['incident_date'] = pd.to_datetime(data['incident_date'])
base_date = data['policy_bind_date'].min()

data['policy_bind_date_diff'] = (data['policy_bind_date'] - base_date).dt.days
data['incident_date_diff'] = (data['incident_date'] - base_date).dt.days
data.drop(['policy_bind_date', 'incident_date', 'policy_id'], axis=1, inplace=True)

# 对所有类别型特征进行标签编码
cat_columns = data.select_dtypes(include='object').columns
for col in cat_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 数据集切分
train = data[data['fraud'].notna()]
test = data[data['fraud'].isna()]

# 初始化并训练模型
model = LGBMClassifier(
    num_leaves=31, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
    max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
    n_estimators=2000, subsample=1, colsample_bytree=1
)
model.fit(train.drop(['fraud'], axis=1), train['fraud'])

# 模型评估
y_pred_proba = model.predict_proba(train.drop(['fraud'], axis=1))[:, 1]
print("Train AUC Score:", roc_auc_score(train['fraud'], y_pred_proba))

# 准备提交文件
test_proba = model.predict_proba(test.drop(['fraud'], axis=1))[:, 1]
submission['fraud'] = test_proba
submission.to_csv('sub.csv', index=False)