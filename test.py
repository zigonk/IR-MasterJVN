from vector_space_model import DOC_DIR, VectorSpaceModel

DOC_DIR = './documents'

VSM = VectorSpaceModel(doc_dir=DOC_DIR)

print(VSM.term_doc_mat)