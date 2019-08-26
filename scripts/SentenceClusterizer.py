import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.decomposition import TruncatedSVD
from hdbscan.hdbscan_ import HDBSCAN



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
    vectorizer = TfidfVectorizer(tokenizer=normalized_tokenizer)
    matrix = vectorizer.fit_transform(texts)
    print("Word Vector Matrix : " + str(matrix.shape))
    decomposer = TruncatedSVD(n_components=dimensions, n_iter=50, random_state=20)
    reduced_matrix = decomposer.fit_transform(matrix)
    print(decomposer.explained_variance_ratio_)
    return reduced_matrix


def hdb_segment(vector_matrix, min_cluster_size=5, min_samples=2, metric="euclidean", cluster_seletion_method="eom"):
    """
    Segments the given data using the HDB clustering algorithm
    :param vector_matrix:
    :param min_cluster_size:
    :param min_samples:
    :param metric:
    :param cluster_seletion_method:
    :return: cluster labels
    """
    hdb_algo = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                       cluster_selection_method=cluster_seletion_method)
    hdb_algo.fit(vector_matrix)

    return hdb_algo.labels_

