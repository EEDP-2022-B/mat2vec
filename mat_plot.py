# Scatter plot to visualize word embeddings using PCA
# Reference: https://builtin.com/machine-learning/nlp-word2vec-python

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

w2v_model = Word2Vec.load(r"mat2vec/training/models/pretrained_embeddings")
words = list(w2v_model.wv.vocab)
# Display the vocabulary words in the console
print(words)

X = w2v_model[w2v_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(w2v_model.wv.vocab)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Scatter Plot of the Word Embeddings")
# Create a scatter plot of the projection of the word embeddings
plt.show()

