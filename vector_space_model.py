import numpy as np
import os
import re
from utils import cos_sim

DOC_DIR = '/home/zigonk/Documents/IR-MasterJVN/documents'

class VectorSpaceModel:
    def __init__(self, doc_dir) -> None:
        self.doc_list = self.load_docs(doc_dir)
        self.term_set = self.build_term_set()
        self.term_doc_mat = self.build_term_doc_mat()
    
    def build_term_doc_mat(self):
        term_doc_mat = np.zeros((len(self.doc_list), len(self.term_set)), dtype=int)
        for ind, doc in enumerate(self.doc_list):
            term_doc_mat[ind] = self.doc2vec(doc)
        return term_doc_mat

    def load_docs(self, doc_dir: str):
        doc_flist = os.listdir(doc_dir)
        doc_list = []
        for fname in doc_flist:
            file = open(os.path.join(doc_dir, fname), 'r')
            data = file.read().rstrip()
            doc_list.append(data)
        return doc_list
    
    def word_tokenize(self, doc: str):
        doc = doc.lower()
        words = [s.strip() for s in re.split(",|;|\n| ", doc)]
        return words


    def build_term_set(self):
        term_set = {}
        ind = 0
        for doc in self.doc_list:
            words = self.word_tokenize(doc)
            for w in words:
                if w not in term_set:
                    term_set[w] = ind
                    ind += 1
        return term_set
    
    def doc2vec(self, doc: str):
        words = self.word_tokenize(doc)
        vec = np.zeros(len(self.term_set), dtype=int)
        for w in words:
            if w in self.term_set:
                vec[self.term_set[w]] += 1
        return vec
    
    def doc_sim(self, doc1, doc2):
        vec1 = self.doc2vec(doc1)
        vec2 = self.doc2vec(doc2)
        return cos_sim(vec1, vec2)