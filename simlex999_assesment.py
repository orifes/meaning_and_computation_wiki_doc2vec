from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import pandas as pd
import pickle as pkl
from scipy import stats

# Loading the models
doc_model = Doc2Vec.load("doc2vec.vec")
word_model = Word2Vec.load("word2vec.model")

# Loading the set of words with both document wnd word vectors
with open("words_we_have.pkl", "rb") as w:
    words = pkl.load(w)

# Loading the simlex dataset
sim_lex_999 = pd.read_csv("SimLex-999.txt", delimiter="\t")

# Finding which of the simlex pairs we cover by our words set
pairs_we_have = sim_lex_999[(sim_lex_999.word2.map(lambda x: x in words) & sim_lex_999.word1.map(lambda x: x in words))]

# Calculating the similarities between each words pair's word and document vectors
doc_sims = []
word_sims = []

for i, r in pairs_we_have.iterrows():
    doc_sims.append(doc_model.docvecs.similarity(r["word1"].capitalize(), r["word2"].capitalize()))
    word_sims.append(word_model.wv.similarity(r["word1"], r["word2"]))

pairs_we_have["doc_vec_sim"] = doc_sims
pairs_we_have["word_vec_sim"] = word_sims

# Calculating the correlation between the word or document embeddings similarities and simlex999 similarity scores
doc_corr, doc_p_val = stats.pearsonr(pairs_we_have["SimLex999"], pairs_we_have["doc_vec_sim"])

word_corr, word_p_val = stats.pearsonr(pairs_we_have["SimLex999"], pairs_we_have["wor_vec_sim"])

# Finding examples of simlex pairs that share a word in order to asses the relative simlex999 similarity and
# word or document similarities

has_more_than_one_appearance = pd.concat([pairs_we_have.word1, pairs_we_have.word2]).value_counts() > 1
num_of_words_with_triplets = has_more_than_one_appearance.sum()
words_with_triplets = list(has_more_than_one_appearance[:num_of_words_with_triplets].index)

# Plotting the "cat" triplet result
print(pairs_we_have[pairs_we_have.word1 == "cat"])
