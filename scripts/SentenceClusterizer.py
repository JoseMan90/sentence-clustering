import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.decomposition import TruncatedSVD
from hdbscan.hdbscan_ import HDBSCAN
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV


def normalized_tokenizer(text):
    """
    Returns the normalized (proprocessed and stemmed) tokens
    :param text: sentence
    :return: list of tokens
    """
    punctuations = dict((ord(char), None) for char in string.punctuation)
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    tokens = nltk.word_tokenize(text.lower().translate(punctuations))
    tokens = [stemmer.stem(item) for item in tokens]
    return tokens


def get_word_vector_matrix(texts, dimensions=10):
    """
    Calculates and returns the reduced words vector matrix
    :param texts: list of sentences
    :param dimensions: dimensions to which the word matrix will be reduced into
    :return: Work vector matrix
    """
    print("Vectorizing sentences into TF-IDF vectors...")
    vectorizer = TfidfVectorizer(tokenizer=normalized_tokenizer)
    matrix = vectorizer.fit_transform(texts)
    print("Word Vector Matrix : " + str(matrix.shape))
    decomposer = TruncatedSVD(n_components=dimensions, n_iter=50, random_state=20)
    reduced_matrix = decomposer.fit_transform(matrix)
    print(decomposer.explained_variance_ratio_)
    return reduced_matrix


def hdb_segment(vector_matrix, min_cluster_size=5, min_samples=2, metric="euclidean", cluster_selection_method="eom"):
    """
    Segments the given data using the HDB clustering algorithm
    :param vector_matrix:
    :param min_cluster_size:
    :param min_samples:
    :param metric:
    :param cluster_seletion_method:
    :return: cluster labels
    """
    print("Running HDB clustering...")

    hdb_algo = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                       cluster_selection_method=cluster_selection_method)
    hdb_algo.fit(vector_matrix)
    scores = pd.DataFrame({"label":hdb_algo.labels_, "probability":hdb_algo.probabilities_})
    scores["confident"] = 0
    scores.loc[scores["probability"]<0.05, "confident"] = 1
    print(scores)
    print(scores["confident"].mean())
    return hdb_algo.labels_


def hdb_scorer(hdb_algo, X):
    """
    Segments the given data using the HDB clustering algorithm
    """
    hdb_algo.fit(X)
    scores = pd.DataFrame({"label":hdb_algo.labels_, "probability":hdb_algo.probabilities_})
    scores["confident"] = 0
    scores.loc[scores["probability"]>=0.05, "confident"] = 1
    scores.loc[scores["label"] == -1, "confident"] = 0
    score = scores["confident"].sum()/scores["label"].count()
    print("Returning score : " + str(score))
    return score


def hdb_segment_generalized(matrix, iterations=50):
    parameter_grid = {
        "min_cluster_size": range(5, 100),
        "min_samples": range(2, 10),
        "metric": ["euclidean"],
        "cluster_selection_method": ["eom", "leaf"],
        "allow_single_cluster": [True, False]
    }
    cv = [(slice(None), slice(None))]
    random_search = RandomizedSearchCV(estimator=HDBSCAN(), param_distributions=parameter_grid,
                      scoring=hdb_scorer, cv=ShuffleSplit(test_size=0.01, n_splits=1), n_jobs=-2, random_state=45,
                                       n_iter=iterations, refit=True)

    random_search.fit(matrix)
    print(random_search.best_score_)
    hdb = random_search.best_estimator_
    print(pd.Series(hdb.labels_).value_counts(normalize=True))
    return hdb.labels_

