from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")


datasets = {
    'en': [
        'mteb/askubuntudupquestions-reranking',
        'mteb/mind_small',
        'mteb/scidocs-reranking',
        'mteb/stackoverflowdupquestions-reranking',
    ],
    'zh': [
        'C-MTEB/CMedQAv1-reranking',
        'C-MTEB/CMedQAv2-reranking',
        'C-MTEB/Mmarco-reranking',
        'C-MTEB/T2Reranking',
    ],
    'fr' : [
        'lyon-nlp/mteb-fr-reranking-alloprof-s2p',
        'lyon-nlp/mteb-fr-reranking-syntec-s2p'
    ]
}
dataset_short_names = {
    'mteb/askubuntudupquestions-reranking': 'AskUbuntu',
    'mteb/mind_small': 'MindSmall',
    'mteb/scidocs-reranking': 'SciDocs',
    'mteb/stackoverflowdupquestions-reranking': 'StackOverflow',
    'C-MTEB/CMedQAv1-reranking': 'CMedQAv1',
    'C-MTEB/CMedQAv2-reranking': 'CMedQAv2',
    'C-MTEB/Mmarco-reranking': 'Mmarco',
    'C-MTEB/T2Reranking': 'T2',
    'lyon-nlp/mteb-fr-reranking-alloprof-s2p': 'Alloprof',
    'lyon-nlp/mteb-fr-reranking-syntec-s2p': 'Syntec',
}
models = {
    'ml': [
        'intfloat/multilingual-e5-small',
        'intfloat/multilingual-e5-base',
        'intfloat/multilingual-e5-large',
    ],
    'en': [
        'mixedbread-ai/mxbai-embed-large-v1',
        'avsolatorio/GIST-large-Embedding-v0',
        'llmrails/ember-v1'
    ],
    'zh': [
        'infgrad/stella-mrl-large-zh-v3.5-1792d',
        'TownsWu/PEG',
        'aspire/acge_text_embedding'
    ],
    'fr': [
        'OrdalieTech/Solon-embeddings-large-0.1',
        'manu/sentence_croissant_alpha_v0.4',
        'manu/bge-m3-custom-fr'
    ]
}


for lang in datasets.keys():
    for dat in datasets[lang]:
        print(f'\n===> {dat}')

        # Load dataset
        dataset = load_dataset(dat)
        try:
            dataset = dataset['test']
        except:
            dataset = dataset['dev']
        
        # Get targets
        targets = [[1] * len(pos) + [0] * len(neg) for pos, neg in zip(dataset['positive'], dataset['negative'])]
        
        for mod in models['ml'] + models[lang]:
            print(f'   => {mod}')
            
            # Initialize embeddings dataset
            dataset_encs = {'query': [], 'documents': [], 'target': targets}
            
            # Load model
            model = SentenceTransformer(mod)

            # Encode queries and documents
            if isinstance(dataset['query'][0], str):
                for query, positive, negative in tqdm(list(zip(dataset['query'], dataset['positive'], dataset['negative']))):
                    dataset_encs['query'].append(model.encode(query).tolist())
                    dataset_encs['documents'].append(model.encode(positive + negative).tolist())
            elif isinstance(dataset['query'][0], list):
                for query, positive, negative in tqdm(list(zip(dataset['query'], dataset['positive'], dataset['negative']))):
                    dataset_encs['query'].append(model.encode(positive + negative).mean(0).tolist())
                    dataset_encs['documents'].append(model.encode(positive + negative).tolist())
    
            # Push dataset to hub
            dat_name = dataset_short_names[dat]
            mod_name = mod.split('/')[1]
            dataset_encs = Dataset.from_dict(dataset_encs)
            dataset_encs.push_to_hub(f'hgissbkh/{dat_name}-{mod_name}-reranking')