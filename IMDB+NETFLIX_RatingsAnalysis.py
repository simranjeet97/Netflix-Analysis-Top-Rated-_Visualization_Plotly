import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

#Make the Ratings DataFrame
title_rating = pd.read_table(r"/content/drive/MyDrive/Colab Notebooks/ratings.tsv", sep='\t')
title_rating.shape
#check if we have unique ratings for the titles
title_rating.groupby(['tconst'], as_index=False).count()

#Maka Title Names DataFrame
title=pd.read_table(r"/content/drive/MyDrive/Colab Notebooks/data.tsv", sep='\t')
title=title.drop_duplicates()
title=title[['titleType','tconst','primaryTitle', 'originalTitle', 'startYear']]
title=title[title.titleType=='movie']
title=title[title.startYear.apply(lambda x: str(x).isnumeric())]
title.head()

#Check the Unique Count
grouped=title.groupby(['primaryTitle', 'startYear'], as_index=False).count()
grouped.head()

#Merger
rt=pd.merge(title_rating.set_index('tconst'), title.set_index('tconst'), left_index=True, right_index=True, how='inner')
rt=rt.drop_duplicates()

#Netflix DataFrame
netflix = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/netflix_titles.csv",index_col="show_id")
netflix.head()

#DropNA for Release Year
netflix=netflix.dropna(subset=['release_year'])
#Change ASTYPE of Relase Year column
netflix.release_year=netflix.release_year.astype(np.int64)
#Drop rows in RT with non-numeric values for startYear and convert to integer.
rt=rt[rt.startYear.apply(lambda x: str(x).isnumeric())]
rt.startYear=rt.startYear.astype(np.int64)
#LowerCase to all the Names
netflix['title']=netflix['title'].str.lower()
rt['originalTitle']=rt['originalTitle'].str.lower()
rt['primaryTitle']=rt['primaryTitle'].str.lower()
#Join Netlfix with IMDB(RT)
netflix=netflix[netflix.type=='Movie']
final_df=pd.merge(netflix, rt, left_on=['title', 'release_year'], right_on=['primaryTitle', 'startYear'], how='inner')
#Now Analyse and Visualise
final_df.sort_values(by=['averageRating', 'numVotes'], inplace=True, ascending=False)
#More then 2000 Votes
final_df_2000=final_df[final_df.numVotes>2000]
final_df_2000.head()

#Visualization of the Destriubutions
plt.figure(figsize=(15, 6))
sns.distplot(final_df['averageRating']);

df = final_df[['numVotes']] #returns a numpy array
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df) 
df.loc[:,:] = scaled_values.round(decimals=6)
norm = pd.DataFrame(columns={'Values','Count'})
norm['Count'] = df['numVotes'].value_counts().values
norm['Values'] = df['numVotes']
norm

#Votes
plt.figure(figsize=(20, 6))
sns.displot(norm,y = norm['Count'],x=norm['Values']);

#Top10 Movies
final_df_2000.head(10)['title']

#TopCountriesProducingMovies
plt.figure(figsize=(20, 6))
chart=sns.countplot(x="country", data=final_df_2000.head(100), order = final_df_2000.head(100)['country'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)

#TopGenres
from itertools import chain

# return list from series of comma-separated strings
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

# calculate lengths of splits
lens = final_df_2000.head(100)['listed_in'].str.split(',').map(len)

# create new dataframe, repeating or chaining as appropriate
res = pd.DataFrame({'title': np.repeat(final_df_2000.head(100)['title'], lens),
                    'listed_in': chainer(final_df_2000.head(100)['listed_in']),
                    })
res['listed_in']=res['listed_in'].str.strip()

print(res)


top_genres=res['listed_in'].value_counts()

top_genres

plt.figure(figsize=(20, 6))
chart=sns.countplot(x="listed_in", data=res, order = res['listed_in'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
