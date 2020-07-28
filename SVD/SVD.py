import pandas as pd
from surprise import SVD, SVDpp, BaselineOnly, Reader, Dataset, accuracy, trainset
from surprise.model_selection import cross_validate, train_test_split
pd.set_option('display.max_columns', None)


def import_data():
    rating_data = pd.io.parsers.read_csv('../dataset/ml-latest-small/ratings.csv',
                                         skiprows=1,
                                         names=['userID', 'movieID', 'rating', 'time'],
                                         engine='python', delimiter=',')
    movies_data = pd.io.parsers.read_csv('../dataset/ml-latest-small/movies.csv',
                                         skiprows=1,
                                         names=['movie_id', 'title', 'genre'],
                                         engine='python', delimiter=',')
    return rating_data, movies_data


def clean_data(rating_data):
    # Delete time column
    del rating_data['time']
    # Check if time column is deleted
    print(rating_data.head())
    # Inspect rating distribution
    data = rating_data['rating'].value_counts().sort_index(ascending=False)
    print(data)
    data.to_csv("rating_distribution.csv", encoding='utf-8', index=True, header=True)
    # Inspect rating distribution by movie
    data = rating_data.groupby('movieID')['rating'].count().clip(upper=50)
    print(data)
    data.to_csv("rating_distribution_movie.csv", encoding='utf-8', index=True, header=True)
    # Inspect rating distribution by movie in order
    data = rating_data.groupby('movieID')['rating'].count().reset_index().sort_values('rating', ascending=False)[:10]
    print(data)
    # Inspect rating distribution by user
    data = rating_data.groupby('userID')['rating'].count().clip(upper=50)
    print(data)
    data.to_csv("rating_distribution_user.csv", encoding='utf-8', index=True, header=True)
    # Inspect rating distribution by user in order
    data = rating_data.groupby('userID')['rating'].count().reset_index().sort_values('rating', ascending=False)[:10]
    print(data)
    # Reduce data size to most relevant users and movies, by selecting users whom have rated 50 movies
    # and movies that have been rated 50 times
    min_movie_ratings = 50
    filter_movies = rating_data['movieID'].value_counts() > min_movie_ratings
    filter_movies = filter_movies[filter_movies].index.tolist()

    min_user_ratings = 50
    filter_users = rating_data['userID'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()

    clean_rating_data = rating_data[
        (rating_data['movieID'].isin(filter_movies)) & (rating_data['userID'].isin(filter_users))]
    clean_rating_data.to_csv("clean_rating_data.csv", encoding='utf-8', index=True, header=True)
    print('The original data frame shape:\t{}'.format(rating_data.shape))
    print('The new data frame shape:\t{}'.format(clean_rating_data.shape))

    watcher = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(clean_rating_data[['userID', 'movieID', 'rating']], watcher)
    return data


def main():
    # Importing data and separating out the movies title from data
    rating_data, movies_data = import_data()
    # Cleaning data to be usable
    data = clean_data(rating_data)
    # Testing which SVD is more accurate using rmse as measurement
    benchmark = []
    # Iterate over SVD, SVD++, BaselineOnly algorithms
    for algorithm in [SVD(), SVDpp(), SVD(biased=False)]:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    # Comparing which algorithm returns the lowest rmse
    test = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    print(test)
    test.to_csv("three_algo.csv", encoding='utf-8', index=True, header=True)

    # SVD++ was chosen as it had the lowest rmse
    print('Using SVDpp')
    algo = SVDpp()
    cross_validate(algo, data, measures=['RMSE'], cv=4, n_jobs=2, verbose=True)

    trainset, testset = train_test_split(data, test_size=0.25)
    algo = SVDpp()
    predictions = algo.fit(trainset).test(testset)
    accuracy.rmse(predictions)

    results = pd.DataFrame(predictions, columns=['userID', 'movieID', 'rating', 'predictedRating', 'details'])
    results.to_csv("results.csv", encoding='utf-8', index=True, header=True)
    results['error'] = abs(results.predictedRating - results.rating)
    best_predictions = results.sort_values(by='error')[:50]
    worst_predictions = results.sort_values(by='error')[-50:]
    best_predictions.to_csv("best_predictions.csv", encoding='utf-8', index=True, header=True)
    worst_predictions.to_csv("worst_predictions.csv", encoding='utf-8', index=True, header=True)


if __name__ == "__main__":
    main()
    pass