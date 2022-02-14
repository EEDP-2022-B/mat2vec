"""
Study on how changing input parameters in the algorithms (e.g., word2vec, phrase2vec) changes the outputs.
"""

# Packages
from gensim.models import Word2Vec
from mat2vec.processing import MaterialsTextProcessor
text_processor = MaterialsTextProcessor()

# Selecting keyword
keyword = "steel"
print("The keyword is", keyword)

# Checking similarities -- pretrained model
w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")
result_pretrain = w2v_model.wv.most_similar(keyword)
print("Pretrained model:", result_pretrain)
print("\tThe pretrained model will be the considered the standard due to the large amount of data it's trained on.")

# Checking similarities -- model trained on corpus example with default settings
w2v_model = Word2Vec.load("mat2vec/training/models/model_default")
result_default = w2v_model.wv.most_similar(keyword)
print("Default model trained on corpus:", result_default)
print("\tThe default model trained on the corpus does not perform as well as the pretrained model. This is because the "
      "corpus is significantly smaller than the amount of data used for the pretrained model.")

# Checking similarities -- model trained on corpus example with 3 epochs
w2v_model = Word2Vec.load("mat2vec/training/models/model_default_3epochs")
result_default_3epochs = w2v_model.wv.most_similar(keyword)
print("Default model with 3 epochs:", result_default_3epochs)
print("\tReducing the number of epochs from 30 (default) to 3 causes a decrease of certainty that a word"
      " is associated with the keyword.")

# Checking similarities -- model trained on corpus example with 300 epochs
w2v_model = Word2Vec.load("mat2vec/training/models/model_default_300epochs")
result_default_300epochs = w2v_model.wv.most_similar(keyword)
print("Default model with 300 epochs:", result_default_300epochs)
print("\tIncreasing the number of epochs to 300 increases the certainty of the association of words and even produces"
      "a result in the 3rd position (i.e., Fe) which is a component of steel.")

# Checking similarities -- model trained on corpus example with size 100
w2v_model = Word2Vec.load("mat2vec/training/models/model_default_100size")
result_default_100size = w2v_model.wv.most_similar(keyword)
print("Default model with size 100:", result_default_100size)
print("\tDecreasing the size from 200 (default) to 100 causes a slight increase in association of a "
      "word with the keyword.")

# Checking similarities -- model trained on corpus example with size 300
w2v_model = Word2Vec.load("mat2vec/training/models/model_default_300size")
result_default_300size = w2v_model.wv.most_similar(keyword)
print("Default model with size 300:", result_default_300size)
print("\tIncreasing the size from 300 causes a slight decrease in association of a word with the keyword.")

# In review
print("In the end, a larger training set is definitely needed to produce results that can be more easily "
      "quantitatively analyzed.")