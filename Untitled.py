#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# In[36]:


get_ipython().run_line_magic('pwd', '')


# In[2]:


data = pd.read_csv('steam-200k.csv')
data.head()


# In[3]:


data_copy = data.copy()
data_copy.head()


# # Edit data

# In[4]:


data = data.rename(columns = {"The Elder Scrolls V Skyrim": "Games"})
data = data.rename(columns = {'151603712' : 'UserID'})
data = data.rename(columns = {'1.0' : 'Hoursplay'})
data = data.rename(columns = {'purchase': 'Action'})

data.head()


# In[5]:


data.isnull().sum()


# # Data discovery

# In[6]:


data.info()


# In[7]:


data.loc[data['Action'] == 'play'].describe()


# In[8]:


values = data.groupby(['UserID', 'Action']).size()
values.head()


# In[9]:


print('Number of games: {0}'.format(len(data.Games.unique())))
print('Number of User: {0}'.format(len(data.UserID.unique())))
print('Number of Hourplay: {0}'.format(len(data.Hoursplay.unique())))
print('Number of Action_purchase: {0}'.format(len(data.loc[data['Action'] == 'purchase'])))
print('Number of Action_play: {0}'.format(len(data.loc[data['Action'] == 'play'])))


# In[7]:


from pandas.api.types import CategoricalDtype
data.Action = data.Action.astype(CategoricalDtype(ordered = True))


# In[8]:


data.dtypes


# In[9]:


data.Action.head()


# In[14]:


data['Name of the games'].value_counts()


# In[10]:


data['Action'].value_counts()


# In[16]:


data.describe().T


# In[18]:


data.columns


# # Visualizations

# In[ ]:


#Đầu tiên sẽ Biểu thị số giờ chơi để xem game được mua nhiều nhất có được chơi nhiều nhất hay không
#Ở đây tạo 2 dataframe mới 'data_purchase' và 'data_play' bằng cách lọc dữ liệu từ data với điều kiện là 'purchase' hoặc 'play'
#Sử dụng 'groupby' để nhóm dữ liệu theo tên trò chơi vầ đếm số lần mua. Sau đó tạo 1 dataframe mới 'purchased_times' để lưu trữ thông tin này và giữ lại số lượng mua cao nhất.
#Tương tự như tính số lần mua cho từng trò chơi, sử dụng 'groupby' để nhóm dữ liệu theo tên trò chơi và tính tổng số giờ chơi. Sau đó vẫn tạo 1 dataframe mới 'hours_played' để lưu trữ thông tin này và giữ lại các giá trị hàng đầu.
#Vẽ biểu đồ:
#Vẽ biểu đồ sử dụng thư viện seaborn để so sánh số lần mua và số giờ chơi cho mỗi trò chơi. Biểu đồ được chia thành 2 cột với số lần mua ở cột đầu tiên và số giờ chơi ở cột hai


# In[40]:


num_games = 10
# Lọc dữ liệu theo hành động
data_purchase = data[data['Action'] == 'purchase']
data_play = data[data['Action'] == 'play']

# Tính toán số lần mua cho từng trò chơi
purchased_times = data_purchase['Games'].value_counts().reset_index()
purchased_times.columns = ['game', 'times_purchased']
purchased_times = purchased_times.head(num_games)

# Tính toán tổng số giờ chơi cho từng trò chơi
hours_played = data_play.groupby('Games')['Hoursplay'].sum().reset_index().sort_values(by = 'Hoursplay', ascending = False)
hours_played.columns = ['game', 'hours_played']
hours_played = hours_played.head(num_games)

# Vẽ biểu đồ
fig, ax = plt.subplots(1, 2, figsize=(12, num_games))

sns.barplot(y='game', x='times_purchased', data=purchased_times, ax=ax[0])
ax[0].set_title('Top Games by Purchase Count')

sns.barplot(y='game', x='hours_played', data=hours_played, ax=ax[1])
ax[1].set_title('Top Games by Total Hours Played')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position('right')

plt.show()


# In[42]:


# Ở biểu đồ phân tích ta nhận thấy game được mua nhiều nhất cũng là game được chơi nhiều nhất, 
# điều này có thể dựa trên sự quan tâm và ưu chuộng của người chơi. Tuy nhiên, khi xem xét trò chơi được mua nhiều thứ hai, 
# chúng ta nhận thấy một điều khác biệt.
# Game được mua nhiều thứ hai không nhất thiết được chơi nhiều nhất, có thế xuất phát từ nhiều yếu tố
# như chất lượng của trò chơi, khả năng giữ chân của nó, hoặc chiến lược tiếp thị. Điều này 
# làm nổi bật sự khác biệt giữa việc mua và việc chơi, cho thấy rằng số lần mua không nhất thiết
# phản ánh độ phổ biến thực sự của một trò chơi trong cộng đồng người chơi.


# In[62]:


# Tiếp theo chúng ta vẽ biểu đồ cột để hiển thị thời lượng chơi cho các game phổ biến nhất giữa 
# top 10 người chơi. Dữ liệu được lấy từ 'data_infos_user', bộ dữ liệu đã được lọc chỉ chứa
# thông tin về các trò chơi được chơi bởi top 10 người chơi. Cột X là thời lượng chơi, Y là game, 
# và biểu đồ thể hiện phân bố thười lượng chơi cho từng trò chơi trong top 10 người chơi.


# In[60]:


top = 10

user_counts = data.groupby('UserID')['Hoursplay'].sum().reset_index().sort_values(by='Hoursplay', ascending=False)[:top]
mask = data['UserID'].isin(user_counts['UserID'])
data_infos_user = data.loc[mask]

hours_played = data_infos_user.groupby('Games')['Hoursplay'].sum().reset_index().sort_values(by='Hoursplay', ascending=False)[:num_games]

sns.barplot(y='Games', x='Hoursplay', data=hours_played)
plt.title('Top 10 Games Played by Top 10 Users')
plt.show()


# In[61]:


# Bằng cách kiểm tra số liệu với những người chơi hàng đầu, chúng ta sẽ thấy một số game như 
# Civilization V và Mount & Blade Warband có sự khác biệt giữa việc được chơi nhiều nhất trong top
# đầu và xuất hiện trong danh sách top 10 trò chơi phổ biến. Điều này cho thấy sự ảnh hưởng
# của việc giữ lại nhừng người chơi tích cực nhất khi phân tích thói quen chơi game.


# In[66]:


#here we compute the number of games a user has bought
user_counts = data_purchase.groupby('UserID')['UserID'].count()

#here we compute the number of hours he has played 
hours_played = data_play.groupby('UserID')['Hoursplay'].sum().sort_values(ascending=False)

#df creation
user_purchased = pd.DataFrame({'UserID': user_counts.index, 'nb_purchased_games': user_counts.values})
user_hours_played = pd.DataFrame({'UserID': hours_played.index, 'hours_played': hours_played.values})

#merge to have one entry per user with number of hours played and number of purchased games
data = pd.merge(user_purchased, user_hours_played, on='UserID')
sns.jointplot(x="nb_purchased_games", y="hours_played", data=data)


# In[19]:


# tần suất xuất hiện của các giá trị trong cột 'purchase' 
data['purchase'].value_counts().plot.barh().set_title('purchase column frequency')


# In[20]:


sns.barplot(x = 'purchase', y = data.purchase.index, data = data)


# In[21]:


data['Hoursplay'].plot()


# In[23]:


# phân phối của Hoursplay, X: khoảng thời gian chơi, y: tần suất
data.Hoursplay.plot(kind = 'hist', bins = 100, figsize = (10,10))
plt.show()


# In[24]:


sns.kdeplot(data.Hoursplay, shade = True, kernel = 'gau')


# In[25]:


# sns.pairplot(data.sample(100), diag_kind = 'hist', kind = 'scatter')
sns.pairplot(data.sample(1000))


# In[26]:


# hue giúp phân biệt giữa các mức độ của biến 'purchase'
sns.pairplot(data.sample(1000), hue = 'purchase')


# In[27]:


sns.pairplot(data.sample(1000), kind = 'reg', hue = 'purchase')


# In[28]:


# hiển thị phân phối của thời gian chơi cho mỗi giá trị trong cột 'purchase'
sns.boxplot(x = 'purchase', y = 'Hoursplay', data = data, palette = 'rainbow')


# In[29]:


# biểu diễn mức độ tương quan giữa các biến trong ma trận, xác định mối quan hệ giữa các biến
# sns.heatmap(data.corr(), annot = True)

plt.figure(figsize = (10,7))
sns.heatmap(data.corr(), annot = True)
plt.title('Correlation between the columns')
plt.show()

