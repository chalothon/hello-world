#---import modules and set up logging
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec
import gensim
import logging
from DocSim import DocSim
import nltk
import os

# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.
#----------------------------------------------------------
#-------------------------Load Word2Vec Model--------------
logging.basicConfig(format='%(asctime)s:%(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load("./data/wiki.en.word2vec_200.model")
stopwords_path = "./data/stopwords_en.txt"


with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

source_doc = "description, flow, symphony, francisco, san, internet, given, no, geometer, sketchpad, easy, across, data, understand,site, this, excel,  even , thinking , student" ## Topic 1 from 900
target_docs = ['learning, online, course, the, web, teaching, faculty, students, this, student, site, strong, study, tool, br, environment, use, management, free, classroom', ##Topic1 from ASS
               "theor, online, music, history, instruction, recorder, analysis, member, teoria, complete, tutorials, play, articles, reference, section, com, exercises, wilson, sonoma, brian", ##Topic1 from ART
               'solving, techniques, problem, complex, tools, problems, collection, advertising, random, thinking, word, literacy, marketing, examples, provided, using, included, excel, ideas, ads'] ##Topic1 from Business

sim_scores = ds.calculate_similarity(source_doc, target_docs)

print(sim_scores )

# Prints:
##   [ {'score': 0.99999994, 'doc': 'delete a invoice'}, 
##   {'score': 0.79869318, 'doc': 'how do i remove an invoice'}, 
##   {'score': 0.71488398, 'doc': 'purge an invoice'} ]
