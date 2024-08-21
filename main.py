import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import nltk
import pickle

movies = pd.read_csv('C:/Users/ansum/Desktop/Ansuman/Data Set/TMDB 5000 movie dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/ansum/Desktop/Ansuman/Data Set/TMDB 5000 movie dataset/tmdb_5000_credits.csv')

movies = movies.merge(credits, on = 'title')
movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.iloc[0]['genres']

def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    
    return l

movies['genres'] = movies['genres'].apply(convert)
movies.head(2)
movies.iloc[0]['keywords']
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter +=1
    return l

movies['cast'] = movies['cast'].apply(convert_cast)


def fetch_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else x)
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

new_df = movies[['id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(i)#new_df.iloc[i[0]].title
    return

print(recommend('Avatar'))
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))










