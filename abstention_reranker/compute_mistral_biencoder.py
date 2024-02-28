import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def compute_document_mistral_scores(queries_pr, positives_pr, negatives_pr, model, tokenizer):

    num_instances = len(queries_pr)
    num_docs_pr = len(positives_pr[0]) + len(negatives_pr[0])
    scores, targets = np.zeros((num_instances, num_docs_pr)), np.zeros((num_instances, num_docs_pr))

    def encode_sample(query, positive, negative):
        # model_name is for caching

        max_length = 1024
        input_texts = [query] + positive + negative
        tot_embeddings = []
        # Tokenize the input texts

        # bs = 2
        for i in range(0, len(input_texts), 2):
            batch_dict = tokenizer(
                input_texts[i : i + 2],
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )
            # append eos_token_id to every input_ids
            batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
            batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")
            # cast to device
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

            # outputs = model(**batch_dict, output_hidden_states=True)
            # embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            tot_embeddings.append(embeddings)

        embeddings = torch.cat(tot_embeddings, dim=0)
        scores = (embeddings[:2] @ embeddings[2:].T) * 100
        print(scores.tolist())

        # scores to numpy
        #scores_instance = scores.numpy()

        #return scores_instance, np.array([1] * len(positive) + [0] * len(negative))
        return scores.tolist(), [1] * len(positive) + [0] * len(negative)

    for i, (query, positive, negative) in tqdm(list(enumerate(zip(queries_pr, positives_pr, negatives_pr)))):
        #scores[i], targets[i] = encode_sample(query, positive, negative)
        sc, tgt = encode_sample(query, positive, negative)
        scores.append(sc)
        targets.append(tgt)

    return scores, targets
