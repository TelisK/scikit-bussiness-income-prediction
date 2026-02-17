import pandas as pd
import files

df = pd.concat([pd.read_csv(f) for f in files.files], ignore_index=True)

df = df.iloc[2:].reset_index(drop=True)
df = df[['Ημερα', 'Έσοδα']]
df = df.dropna()
df['Ημερα'] = pd.to_datetime(df['Ημερα'])
df = df.sort_values('Ημερα')
df = df.reset_index()
print(df.head())