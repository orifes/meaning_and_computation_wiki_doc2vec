from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import pickle as pkl

# Loading the models
doc_model = Doc2Vec.load("doc2vec.vec")
word_model = Word2Vec.load("word2vec.model")

# Creates a list of one word document titles that have a word2vec vector
words = []
word_vecs = []
doc_vecs = []
for doc_title in doc_model.docvecs.offset2doctag:
    try:
        word_vecs.append(word_model.wv.get_vector(doc_title.lower()))
        doc_vecs.append(doc_model.docvecs[doc_title])
        words.append(doc_title.lower())
    except KeyError:
        continue

with open("words_we_have.pkl", "wb") as f:
    pkl.dump(words, f)
