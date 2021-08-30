import requests
import pywikibot
import pickle
from tqdm import tqdm

# Pairing each article title with one not hidden Wikipedia category , that have a short name not containing
# the article title- we do that in order to filter "meta data" categories that each Wiki article have
with open("words_we_have.pkl", "rb") as w:
    words = pickle.load(w)

site = pywikibot.Site("en", "wikipedia")

with open("categories.csv", "w+", encoding="utf-8") as categories_file:
    categories_file.write("word,category\n")
    for word in tqdm(words):
        res = pywikibot.Page(site, word).categories()
        for c in res:
            title = c.title().replace("Category:", "")
            if ("hidden" not in c.categoryinfo) and (word.capitalize() not in title) and (
                    len(title.split(" ")) < 3) and ("disambiguation pages" not in title.lower()):
                categories_file.write(word + "," + title.lower() + "\n")
                break
