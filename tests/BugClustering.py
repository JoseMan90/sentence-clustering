import pandas as pd
from scripts import SentenceClusterizer
from scripts import StringVectorVisualizer

def load_text():
    data = pd.read_csv("../data/1.csv").append(pd.read_csv("../data/2.csv"), ignore_index=True).\
        append(pd.read_csv("../data/3.csv"), ignore_index=True).append(pd.read_csv("../data/4.csv"), ignore_index=True).\
        append(pd.read_csv("../data/5.csv"), ignore_index=True)
    print(data.head())
    print(data.shape)
    return data["Summary"]


if __name__=="__main__":
    messages = load_text()
    matrix = SentenceClusterizer.get_word_vector_matrix(messages)
    StringVectorVisualizer.visualize(matrix)
