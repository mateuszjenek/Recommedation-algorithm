import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data_path = "/Users/mateuszjenek/Desktop/PDB/ratings_Beauty.csv"

df = pd.read_csv(data_path)[0],

usecols = ["UserId", "ProductId", "Rating"],
dtype = {"UserId": "str", "ProductId": "int32", "Rating": "float32"}


df.head()

df_movie_features = df.pivot(
    index="ProductId",
    columns="UserId",
    values="Rating"
).fillna(0)

df_movie_features

num_users = len(df.UserId.unique())
num_items = len(df.ProductId.unique())
print("There are {} unique users and {} unique products in this data set".format(
    num_users, num_items))  # TODO: Formmated string

df_cnt = pd.DataFrame(df.groupby("Rating").size(), columns=["count"])
df_cnt

total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - df.shape[0]
print(rating_zero_cnt)

plt.style.use("ggplot")
get_ipython().run_line_magic("matplotlib", "inline")
ax = df_cnt[["count"]].reset_index().plot(
    x="rating",
    y="count",
    kind="bar",
    figsize=(12, 8),
    title="Count for Each Rating Score",
    logy=False,
    fontsize=12,
)
ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")

# df_movies_cnt = pd.DataFrame(df_ratings.groupby("movieId").size(), columns=["count"])
# df_movies_cnt.head()
# popularity_thres = 50
# popular_movies = list(set(df_movies_cnt.query("count >= @popularity_thres").index))
# df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
# print("shape of original ratings data: ", df_ratings.shape)
# print("shape of ratings data after dropping unpopular movies: ", df_ratings_drop_movies.shape)
# df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby("userId").size(), columns=["count"])
# df_users_cnt.head()
# ratings_thres = 50
# active_users = list(set(df_users_cnt.query("count >= @ratings_thres").index))
# df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
# print("shape of original ratings data: ", df_ratings.shape)
# print("shape of ratings data after dropping both unpopular movies and inactive users: ",
# df_ratings_drop_users.shape)
# movie_user_mat = df_ratings_drop_users.pivot(index="movieId", columns="userId",
# values="rating").fillna(0)
# movie_to_idx = {
# i:movie for i, movie in
# enumerate(list(df_movies.set_index("movieId").loc[movie_user_mat.index].title))
# }
# movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
# model_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)
# def fuzzy_matching(mapper, fav_movie, verbose=True):
# match_tuple = []
# # get match
# for idx, title in mapper.items():
# ratio = fuzz.ratio(title.lower(), fav_movie.lower())
# if ratio >= 60:
# match_tuple.append((title, idx, ratio))
# match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
# if not match_tuple:
# print("Oops! No match is found")
# return
# if verbose:
# print("Found possible matches in our database: {0}\n".format([x[0] for x in match_tuple]))
# return match_tuple[0][1]
# def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
# model_knn.fit(data)
# print("You have input movie:", fav_movie)
# idx = fuzzy_matching(mapper, fav_movie, verbose=True)
# if not idx:
# return
# print("Recommendation system start to make inference")
# print("......\n")
# distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
# raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
# key=lambda x: x[1])[:0:-1]
# print("Recommendations for {}:".format(fav_movie))
# for i, (idx, dist) in enumerate(raw_recommends):
# print("{0}: {1}, with distance of {2}".format(i+1, mapper[idx], dist))
# my_favorite = "Star Wars: Episode IV - A New Hope"
# make_recommendation(
# model_knn = model_knn,
# data = movie_user_mat_sparse,
# fav_movie = my_favorite,
# mapper = movie_to_idx,
# n_recommendations = 10)
