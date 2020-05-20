import spacy

import bridge
import output
import labelling

PERSON = 'PERSON'
LOCATION = 'LOC'
COMMON_WORDS_PATH = './files/common_words.txt'

def get_labeled(res, label):
    rv = dict()
    common_words = bridge.process_common_words(COMMON_WORDS_PATH)

    for key, value in res:
        key_ = key.lower().split('’', 1)[0].split()
        key_ = [item for item in key_ if len(item) > 2 and item not in common_words]
        key_ = ' '.join(key_)

        if value == label and key_:
            rv[key_] = rv.get(key_, 0) + 1
    return rv

def get_entities(texts):
    rv = []
    for text in texts:
        rv += [(entity.text, entity.label_) for entity in nlp(text).ents]
    return rv

def get_mains(occurences, n=1):
    sorted_occurences = sorted(occurences.items(), key=lambda item: item[1], reverse=True)[:n]
    return {key: value for key, value in sorted_occurences}



if __name__ == '__main__':
    input_path = './files/The Final Empire - Brandon Sanderson.txt'
    output_path = './res/res.txt'
    max_len = 1000000
    reducer = bridge.SENTENCE_REDUCER
    n_characters, n_locations = 20, 3

    # nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('./res/complex_model')

    book = bridge.read_book(input_path)
    sections = bridge.reducer_function(reducer, book, max_len=max_len)

    input_path = './files/The Final Empire - Brandon Sanderson.txt'
    book = bridge.read_book(input_path)
    sections += bridge.reducer_function(reducer, book, max_len=max_len)

    input_path = './files/The Final Empire - Brandon Sanderson.txt'
    book = bridge.read_book(input_path)
    sections += bridge.reducer_function(reducer, book, max_len=max_len)


    # Processing entities
    # entities = get_entities(sections)
    # characters = get_labeled(entities, PERSON)
    # locations = get_labeled(entities, LOCATION)

    # main_characters = get_mains(characters, n_characters)
    # main_locations = get_mains(locations, n_locations)
    
    # data = bridge.process_results(output_path)
    # output.print_bar(data)
    # output.print_image(output.word_cloud(data))


    # Getting labels to training data
    # labelling.label_training_data(sections, output_path = './res/training_all.csv')

    
    # Getting categories of the text
    res = list()
    for text in sections:
        cat = nlp(text).cats
        res.append((text, cat))

    bridge.output_results(output_path, bridge.process_classification(res, 0.2, 'ABSTAIN'))
