import spacy
import random

import training_data as td


def create_blank_nlp(labels):
    nlp = spacy.blank('en')
    textcat = nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
            )
    nlp.add_pipe(textcat, last=True)

    for label in labels:
        textcat.add_label('ABSTAIN')
        textcat.add_label(label)

    return nlp, textcat

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

print("Loading data...")
data_path = './res/training_data.txt'
with open(data_path, 'r') as f:
     data = eval(f.read())
     texts, cats = data

n_texts = len(texts)

shuffler = list(zip(texts, cats))
random.shuffle(shuffler)
texts, cats = zip(*shuffler)

train_texts = texts[:int(n_texts * 0.5)]
train_cats = cats[:int(n_texts * 0.5)]
dev_texts = texts[int(n_texts * 0.5):]
dev_cats = cats[int(n_texts * 0.5):]

nlp, textcat = create_blank_nlp(td.CLASSES.values())

print(
    "Using {} examples ({} training, {} evaluation)".format(
        n_texts, len(train_texts), len(dev_texts)
    )
)
train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))



pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
    batch_sizes = spacy.util.compounding(4.0, 32.0, 1.001)
    for i in range(20):
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = spacy.util.minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        print(
            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                losses["textcat"],
                scores["textcat_p"],
                scores["textcat_r"],
                scores["textcat_f"],
            )
        )
    
    output_dir = './res/complete_model'
    nlp.to_disk(output_dir)

