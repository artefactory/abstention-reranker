# run method on several datasets

BIENCODERS = [
    "msmarco-MiniLM-L6-cos-v5",
    "msmarco-distilbert-dot-v5",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-distilbert-cos-v1",
    "multi-qa-mpnet-base-cos-v1",
    "all-MiniLM-L6-v2",
    "all-distilroberta-v1",
    "all-mpnet-base-v2",
    "BAAI/bge-base-en",
    "BAAI/bge-large-en",
    "BAAI/bge-base-en-v1.5",
    "BAAI/llm-embedder",
    "intfloat/multilingual-e5-large",
    'intfloat/multilingual-e5-small',
    'intfloat/multilingual-e5-base',
    "intfloat/e5-small-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "llmrails/ember-v1"
]

CUSTOM_BIENCODERS = [
    "jinaai/jina-embeddings-v2-base-en",
]

MISTRAL_BIENCODERS = [
    "intfloat/e5-mistral-7b-instruct"
]

XENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/qnli-distilroberta-base",
    "cross-encoder/stsb-TinyBERT-L-4",
    "cross-encoder/stsb-distilroberta-base",
    "cross-encoder/quora-distilroberta-base",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-MiniLM2-L6-H768",
    ]

CUSTOM_XENCODERS = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large"
]

DATASETS = [
    'mteb/scidocs-reranking',
    'mteb/stackoverflowdupquestions-reranking',
    "mteb/askubuntudupquestions-reranking",
    # # "mteb/mind_small" --> buggy,
    ]

FRENCH_DATASETS = [
    "OrdalieTech/Ordalie-FR-Reranking-benchmark",
    "lyon-nlp/mteb-fr-reranking-alloprof-s2p",
    # "lyon-nlp/mteb-fr-reranking-syntec-s2p",
    "OrdalieTech/MIRACL-FR-Reranking-benchmark",
    # "sproos/mindsmall-fr"
]

CHINESE_DATASETS = [
    "C-MTEB/CMedQAv1-reranking",
    "C-MTEB/Mmarco-reranking"
]

mistral_instructions = {
    'mteb/askubuntudupquestions-reranking': 'Retrieve duplicate questions from AskUbuntu forum',
    'mteb/mind_small': 'Retrieve relevant news articles based on user browsing history',
    'mteb/scidocs-reranking': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
    'mteb/stackoverflowdupquestions-reranking': 'Retrieve duplicate questions from StackOverflow forum',
}

bge_llm_instructions = {
    "query": "Represent this query for retrieving relevant documents: ",
    "key": "Represent this document for retrieval: ",
}

def prefix_queries(queries, model_name, dataset_name=None):
    # return queries
    if "intfloat" in model_name and not ("mistral" in model_name):
        return ["query: " + query for query in queries]
    if model_name in ["intfloat/e5-mistral-7b-instruct"]:
        if dataset_name and dataset_name in mistral_instructions:
            return [f"Instruct: {mistral_instructions[dataset_name]}\nQuery:" + query for query in queries]
    if model_name == "BAAI/llm-embedder":
        return [bge_llm_instructions["query"] + query for query in queries]
    return queries


def prefix_docs(docs, model_name):
    # return docs
    if model_name in ["BAAI/bge-base-en", "BAAI/bge-large-en", "BAAI/bge-base-en-v1.5"]:
        return [["Represent this sentence for searching relevant passages: " + docn for docn in doc] for doc in docs]
    if model_name in ["intfloat/e5-base-v2", "intfloat/multilingual-e5-large"]:
        return [["passage: " + docn for docn in doc] for doc in docs]
    if model_name == "BAAI/llm-embedder":
        return [[bge_llm_instructions["key"] + docn for docn in doc] for doc in docs]
    return docs
