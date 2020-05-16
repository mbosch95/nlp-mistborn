import re
import snorkel.labeling as snlb
import pandas as pd

import training_data as td


def make_labeling_metals(key, value):
    @snlb.labeling_function(name=f'{key}_function')
    def f(data):
        text = data.text.lower()

        allomancy_pattern = r'|'.join(rf'{i}' for i in td.ALLOMANCY)
        allomancy_regex = re.compile(allomancy_pattern, flags=re.I | re.X)
        combination_pattern = r'|'.join(rf'{key}-*{i}' for i in td.ALLOMANCY)
        combination_regex = re.compile(combination_pattern, flags=re.I | re.X)
        metal_regex = re.compile(rf'\b{key}\b', flags=re.I | re.X)


        if not allomancy_regex.findall(text):
            return td.ABSTAIN

        if metal_regex.findall(text) or combination_regex.findall(text):
            return value

        return td.ABSTAIN
    return f

def make_labeling_users(key, value):
    @snlb.labeling_function(name=f'{key}_function')
    def f(data):
        text = data.text.lower()

        allomancy_pattern = r' | '.join(rf'{i}' for i in td.ALLOMANCY)
        allomancy_regex = re.compile(allomancy_pattern, flags=re.I | re.X)

        if not allomancy_regex.findall(text):
            return td.ABSTAIN

        if key in data.text.lower():
            return value

        return td.ABSTAIN
    return f

lfs = list()

for key, value in td.METALS.items():
    lfs.append(make_labeling_metals(key, value))

for key, value in td.USERS.items():
    lfs.append(make_labeling_users(key, value))

def get_L_train(sections):
    data = {key: td.ABSTAIN for key in td.METALS.values()}
    data['text'] = sections
    data['label'] = False
    df_train = pd.DataFrame(data=data)

    applier = snlb.PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)

    for i, function_results in enumerate(L_train):
        for function_result in function_results:
            if function_result != -1:
                df_train.at[i, function_result] = function_result
                df_train.at[i, 'label'] = True

    return df_train, L_train

def get_analysis(L_train):
    return snlb.LFAnalysis(L=L_train, lfs=lfs).lf_summary()


def generate_training_data(df):
    rv = list()

    for _, row in df.iterrows():
        instance = (row['text'], [(0, len(row['text']), 'ALLOMANCY')])
        rv.append(instance)

    return rv


def test(sections):
    output_path = './res/training.csv'
    output_path = './res/training_data.txt'

    df, res = get_L_train(sections)
    analysis = get_analysis(res)
    res = df.query('label == True')
    rv = generate_training_data(res)
    with open(output_path, 'w') as f:
        f.write(str(rv))
    return rv
    # res.to_csv(path_or_buf=output_path)
