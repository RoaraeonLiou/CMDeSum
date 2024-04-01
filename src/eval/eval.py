from bleu import Bleu, nltk_bleu, corpus_bleu
from rouge import Rouge
from meteor import Meteor


def main(hypotheses_path, references_path, length):
    with open(hypotheses_path, 'r') as r:
        hyp = r.readlines()
        hypotheses = {k: [" ".join(v.strip().lower().split()[:length])] for k, v in enumerate(hyp)}
    with open(references_path, 'r') as r:
        ref = r.readlines()
        references = {k: [v.strip().lower()] for k, v in enumerate(ref)}
    ids = list()
    for k, v in references.items():
        if len(v) == 0:
            ids.append(k)
    for index in ids:
        hypotheses.pop(index)
        references.pop(index)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)
    print("c-Bleu:", _)
    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    return bleu * 100, rouge_l * 100, meteor * 100, ind_bleu, ind_rouge


if __name__ == '__main__':
    hyp_address = '/path/of/beamSearchResult'
    ref_address = '/path/of/groundTruth'
    bleu, rouge_l, meteor, ind_bleu, ind_rouge = main(hyp_address, ref_address, 50)
    print("Bleu: ", bleu)
    print("Meteor: ", meteor)
    print("ROUGe-L: ", rouge_l)
    # print("other: ", )
