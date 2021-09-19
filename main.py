import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style= "darkgrid")
import sqlite3 as sql

#importing data
apple_df = pd.read_csv("APPLE.csv",
                        index_col=0,
                        parse_dates=['Date'])
conn = sql.connect("apple.db")
stock_data = pd.read_sql_query("SELECT * from apple", conn)
conn.close()
print(apple_df.head())
print("--------------------------------------------------------------")
print(stock_data.head())
print("--------------------------------------------------------------")
# Describing the Shape of the Data
print("Apple Stock Details:")
print(apple_df.shape)
print(apple_df.dtypes)
print(apple_df.describe(include="all"))
print("--------------------------- \n")

print("Apple shape: ")
print(stock_data.shape)
print(stock_data.dtypes)
print(stock_data.describe(include="all"))

#Grouping
app_le = apple_df.groupby(['Open', 'High'])['Close'].mean().round(2)
pd.set_option('display.max_rows', None)
print('app_le range')
print(app_le)

#Sorting
ap_st = apple_df.groupby(['Open'])['Low'].mean().sort_values(ascending=False).round(2)
print('Apple\'s open and low')
print(ap_st)

#Replacing and cleaning
apple_df['Close'].fillna("No", inplace = True)
apple_df['Close'].value_counts()
print('Values missing in Cleaning up:')
print(apple_df['Close'].isnull().sum())

#Concatinating
openandclose = pd.concat([ap_st,app_le],axis=1)
print (openandclose.info())

apple_df['Adj Close'].replace(to_replace = ['Adjust Close'],value = 'Adjust Close', inplace = True)
apple_df['Adj Close'].value_counts()
apple_df['Open'].value_counts()

#dropping column
apple_df.drop(columns=['Adj Close'],inplace = True)
Total = apple_df.isnull().sum().sort_values(ascending=False)
Percentage = (Total/apple_df.isnull().count()).sort_values(ascending=False)
null = pd.concat([Total, Percentage], axis=1, keys=['Total','Percentages'])
null.head()
print(null)

#finding avg prices
high_prices = apple_df.loc[:,'High'].to_numpy()
low_prices = apple_df.loc[:,'Low'].to_numpy()
avg_prices = (high_prices + low_prices) / 2.0
print("Size of the data : ", len(avg_prices))


#creating & converting to a numpy array
apple_df["Prediction"] = apple_df[["Close"]].shift()
print(apple_df.head())
print(apple_df.tail())
x = np.array(apple_df.drop(["Prediction"], 1))[:]
print(x)

#creating and converting new y get all target values
y = np.array(apple_df["Prediction"])[:]
print(y)

#Running all dataframe
g = sns.pairplot(apple_df, plot_kws={'color':'green'})
plt.show()

#visualising 3 data graphically
g = sns.pairplot(apple_df[['Open','High','Low']], plot_kws={'color':'#bddc0e'})
plt.show()
g = sns.pairplot(apple_df[['Open','High','Low', 'Close']], hue = 'Close')
plt.show()

#Visualisation via heatmap
ap_st = apple_df[['Open','High','Low', 'Close']].corr(method ='kendall')
cols = ['Open ap','High ap','Low ap']
ax = sns.heatmap(ap_st, annot=True,
                 yticklabels=cols,
                 xticklabels=cols,
                 annot_kws={'size': 12})


#investigating apple closing stock
sns.set()
plt.figure(figsize=(10, 4))
plt.title("Apple's Stock")
plt.xlabel("Days")
plt.ylabel("Closing Price USD ($)")
plt.plot(apple_df["Close"])
plt.show()

apple_df = apple_df[["Close"]]
print(apple_df.head())

