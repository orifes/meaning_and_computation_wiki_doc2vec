from gensim.corpora import WikiCorpus, MmCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import get_tmpfile

# Loading the corpus from the wiki dump file
corpus_path = get_tmpfile("wiki-corpus.mm")
wiki = WikiCorpus("enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2")
MmCorpus.serialize(corpus_path, wiki)


# Iterators over the corpus to be used for training the model
class WikiLines(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield content


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c for c in content], [title])


documents = TaggedWikiDocument(wiki)
lines = WikiLines(wiki)

# Training and saving the doc2vec model

doc_model = Doc2Vec(min_count=0)
doc_model.build_vocab(documents)
doc_model.train(documents, total_examples=doc_model.corpus_count, epochs=3)
doc_model.save("doc2vec.model")

# Training and saving the word2vec model

word_model = Word2Vec(min_count=0)
word_model.build_vocab(documents)
word_model.train(documents, total_examples=word_model.corpus_count, epochs=3)
word_model.save("word2vec.model")

