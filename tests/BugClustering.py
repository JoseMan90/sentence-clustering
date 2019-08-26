import pandas as pd
from scripts import SentenceClusterizer
from scripts import StringVectorVisualizer
import numpy as np

def load_text():
    data = pd.read_csv("../data/1.csv").append(pd.read_csv("../data/2.csv"), ignore_index=True).\
        append(pd.read_csv("../data/3.csv"), ignore_index=True).append(pd.read_csv("../data/4.csv"), ignore_index=True).\
        append(pd.read_csv("../data/5.csv"), ignore_index=True)
    print(data.head())
    print(data.shape)
    return data["Summary"].astype(str)


if __name__=="__main__":
    np.random.seed(456)
    messages = load_text()
    matrix = SentenceClusterizer.get_word_vector_matrix(messages)
    labels = SentenceClusterizer.hdb_segment_generalized(matrix, iterations=100)
    StringVectorVisualizer.visualize(matrix, labels=labels, title="Bug descriptions visualized in 2D space using tSNE")
