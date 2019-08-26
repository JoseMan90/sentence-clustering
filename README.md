# sentence-clustering

The module exposes APIs for end to end sentence clutering.

Sentence clustering is done by first converting nlp-preprocessed sentences TF-IDF word vectors. Then reducing the dimensionality using SVD and then applying the HDB algorithm. The HDB algorithm is automatically tuned using a custom scoring function as described in https://towardsdatascience.com/how-to-cluster-in-high-dimensions-4ef693bacc6

The module also visualizes the sentences in 2 dimensions before and after clusterin using the TSNE algorithm.

Future score includes adding tags to each cluster to aid in analysis iof the clusters
