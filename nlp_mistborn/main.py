import en_core_web_sm

from bridge import RECURSIVE_REDUCER, SENTENCE_REDUCER, reducer_function, read_book, output_results, process_common_words


PERSON = 'PERSON'
COMMON_WORDS_PATH = './files/common_words.txt'

def get_labeled(res, label):
    rv = dict()
    common_words = process_common_words(COMMON_WORDS_PATH)

    for key, value in res:
        # Get all names to lower and remove everything after a "´" character
        key_ = key.lower().split('’', 1)[0].split()
        # Remove every word with less than 3 characters from the name of the entity and remmove common words
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
    output_path = './nlp_mistborn/res.txt'
    max_len = 1000000
    reducer = RECURSIVE_REDUCER
    n_characters = 20

    nlp = en_core_web_sm.load()


    book = read_book(input_path)
    sections = reducer_function(reducer, book, max_len=max_len)

    entities = get_entities(sections)
    characters = get_labeled(entities, PERSON)
    main_characters = get_mains(characters, n_characters)

    output_results(output_path, characters)
