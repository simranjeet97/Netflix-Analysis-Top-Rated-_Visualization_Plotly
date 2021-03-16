'''
#Recommendation System
GOAL -
1. Based on Description
'''
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

netflix = pd.read_csv(r"/content/drive/MyDrive/Colab Notebooks/netflix_titles.csv",index_col="show_id")
netflix.head()

movies_des = netflix[netflix['type'] == 'Movie'].reset_index()
movies_des = movies_des[['title', 'description']]
movies_des.head()

tv_des = netflix[netflix['type'] == 'TV Show'].reset_index()
tv_des = tv_des[['title', 'description']]
tv_des.head()

filtered_movies = []
movies_words = []

for text in movies_des['description']:
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
    movies_words.append(tokens_without_sw)
    filtered = (" ").join(tokens_without_sw)
    filtered_movies.append(filtered)

movies_words = [val for sublist in movies_words for val in sublist]
movies_words = sorted(set(movies_words))
movies_des['description_filtered'] = filtered_movies
movies_des.head()

filtered_tv = []
tv_words = []

for text in tv_des['description']:
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
    tv_words.append(tokens_without_sw)
    filtered = (" ").join(tokens_without_sw)
    filtered_tv.append(filtered)

tv_words = [val for sublist in tv_words for val in sublist]
tv_words = sorted(set(tv_words))
tv_des['description_filtered'] = filtered_tv
tv_des.head()

movie_word_binary = [[0] * 0 for i in range(len(set(movies_words)))]

for des in movies_des['description_filtered']:
    k = 0
    for word in movies_words:
        if word in des:
            movie_word_binary[k].append(1.0)
        else:
            movie_word_binary[k].append(0.0)
        k+=1
        
movie_word_binary = pd.DataFrame(movie_word_binary).transpose()


for des in tv_des['description_filtered']:
    k = 0
    for word in tv_words:
        if word in des:
            tv_word_binary[k].append(1.0)
        else:
            tv_word_binary[k].append(0.0)
        k+=1
        
tv_word_binary = pd.DataFrame(tv_word_binary).transpose()



def recommender2(search):
    cs_list = []
    binary_list = []
    if search in movies_des['title'].values:
        idx = movies_des[movies_des['title'] == search].index.item()
        for i in movie_word_binary.iloc[idx]:
            binary_list.append(i)
        point1 = np.array(binary_list).reshape(1, -1)
        point1 = [val for sublist in point1 for val in sublist]    
        for j in range(len(movies_des)):
            binary_list2 = []
            for k in movie_word_binary.iloc[j]:
                binary_list2.append(k)
            point2 = np.array(binary_list2).reshape(1, -1)
            point2 = [val for sublist in point2 for val in sublist]
            dot_product = np.dot(point1, point2)
            norm_1 = np.linalg.norm(point1)
            norm_2 = np.linalg.norm(point2)
            cos_sim = dot_product / (norm_1 * norm_2)
            cs_list.append(cos_sim)
        movies_copy = movies_des.copy()
        movies_copy['cos_sim'] = cs_list
        results = movies_copy.sort_values('cos_sim', ascending=False)
        results = results[results['title'] != search]    
        top_results = results.head(5)
        return(top_results)
    elif search in tv_des['title'].values:
        idx = tv_des[tv_des['title'] == search].index.item()
        for i in tv_word_binary.iloc[idx]:
            binary_list.append(i)
        point1 = np.array(binary_list).reshape(1, -1)
        point1 = [val for sublist in point1 for val in sublist]
        for j in range(len(tv)):
            binary_list2 = []
            for k in tv_word_binary.iloc[j]:
                binary_list2.append(k)
            point2 = np.array(binary_list2).reshape(1, -1)
            point2 = [val for sublist in point2 for val in sublist]
            dot_product = np.dot(point1, point2)
            norm_1 = np.linalg.norm(point1)
            norm_2 = np.linalg.norm(point2)
            cos_sim = dot_product / (norm_1 * norm_2)
            cs_list.append(cos_sim)
        tv_copy = tv_des.copy()
        tv_copy['cos_sim'] = cs_list
        results = tv_copy.sort_values('cos_sim', ascending=False)
        results = results[results['title'] != search]    
        top_results = results.head(5)
        return(top_results)
    else:
        return("Title not in dataset. Please check spelling.")

    pd.options.display.max_colwidth = 300
recommender2('The Conjuring')
