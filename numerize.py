import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('adultIncome.csv')
df.replace(inplace=True, to_replace=[' Never-worked'], value=0)
df.replace(inplace=True, to_replace=[' Private'], value=1)
df.replace(inplace=True, to_replace=[' Local-gov'], value=2)
df.replace(inplace=True, to_replace=[' Federal-gov'], value=3)

df.replace(inplace=True, to_replace=[' Preschool'], value=0)
df.replace(inplace=True, to_replace=[' HS-grad'], value=1)
df.replace(inplace=True, to_replace=[' Some-college'], value=2)
df.replace(inplace=True, to_replace=[' Bachelors'], value=3)
df.replace(inplace=True, to_replace=[' Prof-school'], value=4)
df.replace(inplace=True, to_replace=[' Masters'], value=5)
df.replace(inplace=True, to_replace=[' Doctorate'], value=6)
df.replace(inplace=True, to_replace=[' Never-married'], value=0)
df.replace(inplace=True, to_replace=['Married'], value=1)
df.replace(inplace=True, to_replace=[' Divorced'], value=2)
df.replace(inplace=True, to_replace=[' Widowed'], value=3)
df.replace(inplace=True, to_replace=[' Black'], value=0)
df.replace(inplace=True, to_replace=[' White'], value=1)
df.replace(inplace=True, to_replace=[' Asian-Pac-Islander'], value=2)
df.replace(inplace=True, to_replace=[' Amer-Indian-Eskimo'], value=3)
df.replace(inplace=True, to_replace=[' Other'], value=4)
df.replace(inplace=True, to_replace=[' Male'], value=0)
df.replace(inplace=True, to_replace=[' Female'], value=1)
df.replace(inplace=True, to_replace=[' <=50K'], value=0)
df.replace(inplace=True, to_replace=[' >50K'], value=1)

grouped = df.groupby(df.education)
frames = []
frames.append(grouped.get_group(0))
frames.append(grouped.get_group(1))
frames.append(grouped.get_group(2))
frames.append(grouped.get_group(3))
frames.append(grouped.get_group(4))
frames.append(grouped.get_group(5))
frames.append(grouped.get_group(6))

df.to_csv('incomeProcessed.csv', index=False)

print(normalized_df.head())
