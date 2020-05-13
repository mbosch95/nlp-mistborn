from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
import pandas
import re
import training_data as td


@labeling_function()
def metals(data):
    text = data.text.lower()
    allomancy_pattern = r' | '.join(rf'{i}' for i in td.ALLOMANCY)
    allomancy_regex = re.compile(allomancy_pattern, flags=re.I | re.X)
    
    if not allomancy_regex.findall(text):
        return td.ABSTAIN

    for key, value in td.METALS.items():
        metal_regex = re.compile(rf'\b{key}\b', flags=re.I | re.X)
        if metal_regex.findall(text):
            return value
    
    return td.ABSTAIN

@labeling_function()
def users(data):
    text = data.text.lower()
    allomancy_pattern = r' | '.join(rf'{i}' for i in td.ALLOMANCY)
    allomancy_regex = re.compile(allomancy_pattern, flags=re.I | re.X)
    
    if not allomancy_regex.findall(text):
        return td.ABSTAIN

    for key, value in td.USERS.items():
        if key in data.text.lower():
            return value
    return td.ABSTAIN


lfs = [metals, users]

def get_L_train(sections):
    df_train = pandas.DataFrame(data={
        'text': sections,
        'label': td.ABSTAIN,
    })

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)

    for i, value in enumerate(L_train):
        if value[0] == -1 and value[1] != -1:
            df_train.at[i, 'label'] = value[1]
        else:
            df_train.at[i, 'label'] = value[0]

    return df_train, L_train

def get_analysis(L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_summary()

def test(sections):
    output_path = './res/training.csv'
    df, res = get_L_train(sections)
    analysis = get_analysis(res)
    res = df.query('label != -1')
    res.to_csv(path_or_buf=output_path)
