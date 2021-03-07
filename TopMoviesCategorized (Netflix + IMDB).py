#!pip install pycountry_convert
"""
#HIGH RATED MOVIES
- by the gender of users
- by the age group of users
- by genre
- by country
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2

netflix = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/netflix_titles.csv",index_col="show_id")
netflix.head()
movies = netflix[netflix['type'] == 'Movie']
movies.head()

df = movies[['title', 'release_year', 'listed_in', 'country']]
df = df.rename(columns={'release_year': 'year'})
df.head()

imdb_movies = pd.read_csv(r'/content/drive/MyDrive/Colab Notebooks/IMDb movies.csv')
imdb_movies = imdb_movies[['imdb_title_id', 'title', 'year']]
imdb_ratings = pd.read_csv(r'/content/drive/MyDrive/Colab Notebooks/IMDb ratings.csv')
imdb = pd.merge(imdb_movies, imdb_ratings, on='imdb_title_id')
imdb.head()

imdb.info()

#Remove Unwanted Columns
imdb = imdb[['title', 'year', 'weighted_average_vote', 'allgenders_0age_avg_vote', 'allgenders_0age_votes', 'allgenders_18age_avg_vote', 'allgenders_18age_votes',
             'allgenders_30age_avg_vote', 'allgenders_30age_votes', 'allgenders_45age_avg_vote', 'allgenders_45age_votes',
             'males_allages_avg_vote', 'males_allages_votes', 'males_0age_avg_vote', 'males_0age_votes',
             'males_18age_avg_vote', 'males_18age_votes', 'males_30age_avg_vote', 'males_30age_votes', 
             'males_45age_avg_vote', 'males_45age_votes', 'females_allages_avg_vote', 'females_allages_votes',
             'females_0age_avg_vote', 'females_0age_votes', 'females_18age_avg_vote', 'females_18age_votes',
             'females_30age_avg_vote', 'females_30age_votes', 'females_45age_avg_vote', 'females_45age_votes']]
imdb = imdb.fillna(0)
imdb.head()

#To Check the Data
imdb.describe()

#Real IMDB Weightage Algorithm
##How does IMDB calculate the weighted average?

"""IMDB uses the following equation to calculate the weighted averages:

     wr=(v/v+m)R + (m/v+m)C
 
where

v = number of votes on the movie

m = minimum number of votes to be in the top 50 (currently 1000 votes)

R = average/mean rating for the movie

C = mean rating across whole dataset (5.9 as seen above)

Thus, we will calculate these weighted averages and replace the relevant columns."""

def weighted_averages(number, avg, row):
    if row[number] != 0.0:
        wr = ((row[number]/(row[number]+1000))*row[avg]) + ((1000/(row[number]+1000))*5.9)
    else:
        wr = 0
    return wr


weighted_allgenders_0age = []
weighted_allgenders_18age = []
weighted_allgenders_30age = []
weighted_allgenders_45age = []
weighted_males_allages = []
weighted_males_0age = []
weighted_males_18age = []
weighted_males_30age = []
weighted_males_45age = []
weighted_females_allages = []
weighted_females_0age = []
weighted_females_18age = []
weighted_females_30age = []
weighted_females_45age = []

for i in range(len(imdb)):
    weighted_allgenders_0age.append(weighted_averages('allgenders_0age_votes', 'allgenders_0age_avg_vote', imdb.iloc[i]))
    weighted_allgenders_18age.append(weighted_averages('allgenders_18age_votes', 'allgenders_18age_avg_vote', imdb.iloc[i]))
    weighted_allgenders_30age.append(weighted_averages('allgenders_30age_votes', 'allgenders_30age_avg_vote', imdb.iloc[i]))
    weighted_allgenders_45age.append(weighted_averages('allgenders_45age_votes', 'allgenders_45age_avg_vote', imdb.iloc[i]))
    weighted_males_allages.append(weighted_averages('males_allages_votes', 'males_allages_avg_vote', imdb.iloc[i]))
    weighted_males_0age.append(weighted_averages('males_0age_votes', 'males_0age_avg_vote', imdb.iloc[i]))
    weighted_males_18age.append(weighted_averages('males_18age_votes', 'males_18age_avg_vote', imdb.iloc[i]))
    weighted_males_30age.append(weighted_averages('males_30age_votes', 'males_30age_avg_vote', imdb.iloc[i]))
    weighted_males_45age.append(weighted_averages('males_45age_votes', 'males_45age_avg_vote', imdb.iloc[i]))
    weighted_females_allages.append(weighted_averages('females_allages_votes', 'females_allages_avg_vote', imdb.iloc[i]))
    weighted_females_0age.append(weighted_averages('females_0age_votes', 'females_0age_avg_vote', imdb.iloc[i]))
    weighted_females_18age.append(weighted_averages('females_18age_votes', 'females_18age_avg_vote', imdb.iloc[i]))
    weighted_females_30age.append(weighted_averages('females_30age_votes', 'females_30age_avg_vote', imdb.iloc[i]))
    weighted_females_45age.append(weighted_averages('females_45age_votes', 'females_45age_avg_vote', imdb.iloc[i]))
    
imdb['weighted_allgenders_0age'] = weighted_allgenders_0age
imdb['weighted_allgenders_18age'] = weighted_allgenders_18age
imdb['weighted_allgenders_30age'] = weighted_allgenders_30age
imdb['weighted_allgenders_45age'] = weighted_allgenders_45age
imdb['weighted_males_allages'] = weighted_males_allages
imdb['weighted_males_0age'] = weighted_males_0age
imdb['weighted_males_18age'] = weighted_males_18age
imdb['weighted_males_30age'] = weighted_males_30age
imdb['weighted_males_45age'] = weighted_males_45age
imdb['weighted_females_allages'] = weighted_females_allages
imdb['weighted_females_0age'] = weighted_females_0age
imdb['weighted_females_18age'] = weighted_females_18age
imdb['weighted_females_30age'] = weighted_females_30age
imdb['weighted_females_45age'] = weighted_females_45age

imdb = imdb[['title', 'year', 'weighted_average_vote', 'weighted_allgenders_0age', 'weighted_allgenders_18age', 'weighted_allgenders_30age', 'weighted_allgenders_45age',
            'weighted_males_allages', 'weighted_males_0age', 'weighted_males_18age', 'weighted_males_30age', 'weighted_males_45age',
            'weighted_females_allages', 'weighted_females_0age', 'weighted_females_18age', 'weighted_females_30age', 'weighted_females_45age']]

imdb.head()

#Merge the Netflix and IMDB Dataset
merged = df.merge(imdb, how="inner", left_on=['title', 'year'], right_on=['title', 'year'])
merged.head()
merged.info()

#Check Country Column for Null and then Fill it
merged[merged['country'].isnull()]
merged.loc[6, 'country'] = 'India'
merged.loc[74, 'country'] = 'United Kingdom'
merged.loc[260, 'country'] = 'India'
merged.loc[506, 'country'] = 'Indonesia'

#Top Movies by Age and Gender groups
def top10movies(column, group):
    titles = []
    scores = []
    top10=merged[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    sns.set(rc={'figure.figsize':(12,10)})
    ax = sns.barplot(x=titles, y=scores, data=merged, palette=("crest_d"))
    ticks = ax.set_xticklabels(titles, rotation=90, size = 12)
    ticks = ax.set_yticklabels(ax.get_yticks(), size = 12)
    ax.set(xlabel='Movie Title', ylabel='Mean Rating')
    ax.set_title('Best Rated Movies on Netflix' + group, size= 20)
    return ax


top10movies('weighted_average_vote', ' Overall')
top10movies('weighted_allgenders_0age', ' by Under 18s - Both Genders')
top10movies('weighted_allgenders_18age', ' by Users Aged 18-30')

#Top Movies by Genre
def genres10(genre_string, column, group):
    titles = []
    scores = []
    genre = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        if genre_string in row['listed_in']:
            genre.append(row)
    genre = pd.DataFrame(genre)
    top10=genre[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    sns.set(rc={'figure.figsize':(12,10)})
    ax = sns.barplot(x=titles, y=scores, palette=("crest_d"))
    ticks = ax.set_xticklabels(titles, rotation=90, size = 12)
    ticks = ax.set_yticklabels(ax.get_yticks(), size = 12)
    ax.set(xlabel='Movie Title', ylabel='Mean Rating')
    ax.set_title('Best Rated ' + genre_string + ' on Netflix' + group, size= 20)
    return ax

#Horror
genres10('Horror', 'weighted_average_vote', ' Overall')
genres10('Comedies', 'weighted_average_vote', ' Overall')
genres10('Children & Family Movies', 'weighted_average_vote', ' Overall')
genres10('Romantic Movies', 'weighted_average_vote', ' Overall')

#Distribution of Ratings for Different Genres
subframe = []
first_genre_list = []
for i in range(len(merged)):
    row = merged.iloc[i]
    if 'Horror' in row['listed_in']:
        first_genre_list.append('Horror')
        subframe.append(row)
    elif 'Comedies' in row['listed_in']:
        first_genre_list.append('Comedy')
        subframe.append(row)
    elif 'Children & Family Movies' in row['listed_in']:
        first_genre_list.append('Family')
        subframe.append(row)
    elif 'Romantic Movies' in row['listed_in']:
        first_genre_list.append('Romance')
        subframe.append(row)
    elif 'Action' in row['listed_in']:
        first_genre_list.append('Action')
        subframe.append(row)

subframe = pd.DataFrame(subframe)
subframe['first_genre'] = first_genre_list

sns.displot(subframe, x="weighted_average_vote", hue="first_genre")


#Top Movies by Continent
for i in range(len(merged['country'])):
    split = merged['country'].iloc[i].split(',')
    merged['country'].iloc[i] = split[0]
    if merged['country'].iloc[i] == 'Soviet Union':
        merged['country'].iloc[i] = 'Russia'

continents = {
    'AF': 'Africa',
    'AS': 'Asia',
    'OC': 'Australia',
    'EU': 'Europe',
    'NA': 'North America',
    'SA': 'South America'
}
countries = merged['country']

merged['continent'] = [continents[country_alpha2_to_continent_code(country_name_to_country_alpha2(country))] for country in countries]



data = merged['continent'].value_counts(normalize=True) * 100
labels = data.keys()

pie, ax = plt.subplots(figsize=[12,12])
patches, texts, autotexts = plt.pie(x=data, autopct="%.1f%%", pctdistance=0.9, labels=labels, textprops={'fontsize': 16}, shadow=True, explode=[0,0.1,0,0,0,0], colors=['#5cb68b','#0886ad','#7cbbdf','#48b0df','#5c936a','#5c9ac1'])
plt.title("Movies from each Continent", fontsize=20);


def continent10(continent_string, column, group):
    titles = []
    scores = []
    continent = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        if row['continent'] == continent_string:
            continent.append(row)
    continent = pd.DataFrame(continent)
    top10=continent[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    sns.set(rc={'figure.figsize':(12,10)})
    ax = sns.barplot(x=titles, y=scores, palette=("crest_d"))
    ticks = ax.set_xticklabels(titles, rotation=90, size = 12)
    ticks = ax.set_yticklabels(ax.get_yticks(), size = 12)
    ax.set(xlabel='Movie Title', ylabel='Mean Rating')
    ax.set_title('Best Rated Movies from ' + continent_string + ' on Netflix' + group, size= 20)
    return ax

continent10('Asia', 'weighted_average_vote', ' Overall')

