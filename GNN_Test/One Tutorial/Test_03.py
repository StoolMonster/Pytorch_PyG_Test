from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('../input/yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']

buy_df = pd.read_csv('../input/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
df.head()
