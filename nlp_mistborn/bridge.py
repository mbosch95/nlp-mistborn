from pprint import PrettyPrinter
from nltk.tokenize import sent_tokenize
from ast import literal_eval


SENTENCE_REDUCER = 'sentences'
RECURSIVE_REDUCER = 'fit'

def read_book(path):
    rv = ''
    with open(path, 'r+') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            rv += line
    return rv

def process_common_words(path):
    with open(path, 'r+') as f:
        rv = set(f.read().splitlines())
    return rv

def output_results(path, res):
    prettyfier = PrettyPrinter(width=1)
    with open(path, 'w') as f:
        f.write(prettyfier.pformat(res))

def process_results(path):
    file = open(path, "r")
    contents = file.read()
    return literal_eval(contents)


def reducer_function(func, text, **kwargs):
    reducer = {
        SENTENCE_REDUCER: reduce_to_sentences,
        RECURSIVE_REDUCER: reduce_to_fit,
    }
    return reducer[func](text, **kwargs)

def reduce_to_fit(text, **kwargs):
    max_len = kwargs.get('max_len', 1000000)
    if len(text) > max_len:
        return reduce_to_fit(text[:len(text) // 2], max_len=max_len) + reduce_to_fit(text[len(text) // 2:], max_len=max_len)
    return [text]

def reduce_to_sentences(text, **kwargs):
    return sent_tokenize(text)
