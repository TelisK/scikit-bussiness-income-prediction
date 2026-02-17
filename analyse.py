import pandas as pd
import files
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.concat([pd.read_csv(f) for f in files.files], ignore_index=True)

df = df.iloc[2:].reset_index(drop=True)
df = df[['Ημερα', 'Έσοδα']]
df = df.dropna()
df['Ημερα'] = pd.to_datetime(df['Ημερα'])
df = df.sort_values('Ημερα')
df = df.reset_index()

df['Έσοδα'] = df['Έσοδα'].str.replace('€','')
df['Έσοδα'] = df['Έσοδα'].str.replace(',','')
df['Έσοδα'] = df['Έσοδα'].astype(float)
df = df.drop('index', axis=1)

# Date Features
df['Year'] = df['Ημερα'].dt.year
df['Month'] = df['Ημερα'].dt.month
df['Day'] = df['Ημερα'].dt.day
df['Day_of_week'] = df['Ημερα'].dt.dayofweek
df['Day_of_year'] = df['Ημερα'].dt.dayofyear
df['Weekend'] = (df['Day_of_week'] >= 5).astype(int)

# Lags will be a problem if i want to predict the year income
'''# Lags
df['lag_1_day'] = df['Έσοδα'].shift(1)
df['lag_7_days'] = df['Έσοδα'].shift(7)
df['mean_7_days'] = df['Έσοδα'].rolling(window=7).mean()
'''

X = df.drop(['Έσοδα', 'Ημερα'], axis=1)
y = df['Έσοδα']

date_split = int(len(df) * 0.8)
X_train = X[:date_split]
X_test = X[date_split:]
y_train = y[:date_split]
y_test = y[date_split:]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print('Model Trained!')

y_predict = model.predict(X_test)

mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'Mean Absolute Error: {mae:.2f} €')
print(f'R2 Score: {r2:.2f}')
