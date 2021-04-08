import spacy
from spacy.matcher import Matcher
import pandas as pd
import re


df = pd.read_json(r'/home/orangefiish/PycharmProjects/NLP_HW2_wikiBioCorpus/ppl_bio.json')
df.columns = ['Name', 'Biography']
print('/Raw Data/\n', df.head())
print(df.shape, '\n')


# preprocess biography data
def clean(text):
    # remove new line characters
    text = re.sub('\n ', '', str(text))
    text = re.sub('\n', ' ', str(text))
    # remove apostrophes
    text = re.sub("'s", '', str(text))
    # remove quotation marks
    text = re.sub('\"', '', str(text))

    return text


# add cleaned biographies to dataframe
df['Biography_clean'] = df['Biography'].apply(clean)


# split sentences
def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent


# add split sentences to dataframe
df['Split_Sent'] = df['Biography_clean'].apply(sentences)


# create a dataframe containing sentences
df_sents = pd.DataFrame(columns=['Name', 'Sent'])
# list of sentences for new df
row_list = []

# go over the biographies in df
for i in range(len(df)):
    # go over the sentences in the biography
    for sent in df.loc[i, 'Split_Sent']:
        # name
        name = df.loc[i, 'Name']
        # create a dictionary and write it to list
        dict_sents = {'Name': name, 'Sent': sent}
        row_list.append(dict_sents)

# create the new dataframe
df_sents = pd.DataFrame(row_list)
print('/Preprocessed Data/\n', df_sents.head())
print(df_sents.shape, '\n')

# load english language model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


# write rules to find birth places
def find_birth_place(text):
    birth_places = []

    # create a spacy doc
    doc = nlp(text)

    # define the pattern
    pattern = [{'LOWER': 'born'},
               {'LOWER': 'in', 'OP': '?'},
               {'POS': 'PROPN'}]

    pattern1 = [{'ORTH': 'in'},
                {'POS': 'PROPN'},
                {'ORTH': '–'}]

    # match class object
    matcher = Matcher(nlp.vocab, validate=True)
    matcher.add("birth_place", None, pattern, pattern1)

    matches = matcher(doc)

    # find patterns in the text
    for i in range(0, len(matches)):
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        birth_places.append(str(token))

    return birth_places


# apply function
df_sents['Place_of_Birth'] = df_sents['Sent'].apply(find_birth_place)

# count the matched sentences
count=0
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Place_of_Birth']) != 0:
        count += 1

print('\n')
print('/', count, ' birth places found/')


# check if each matched sentence belongs to different person
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Place_of_Birth']) != 0:
        print(df_sents.loc[i, 'Name'], '->', df_sents.loc[i, 'Place_of_Birth'], '\n')


# write rules to find birth years
def find_birth_year(text):
    birth_years = []

    # create a spacy doc
    doc = nlp(text)

    # define the pattern
    pattern = [{'ORTH': '('},
               {'LOWER': 'born', 'OP': '?'},
               {'SHAPE': 'dd'},
               {'POS': 'PROPN'},
               {'SHAPE': 'dddd'}]

    # match class object
    matcher = Matcher(nlp.vocab, validate=True)
    matcher.add("birth_year", None, pattern)

    matches = matcher(doc)

    # find patterns in the text
    for i in range(0, len(matches)):
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        birth_years.append(str(token))

    return birth_years


# apply function
df_sents['Year_of_Birth'] = df_sents['Sent'].apply(find_birth_year)

# count the matched sentences
count=0
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Year_of_Birth']) != 0:
        count += 1

print('\n')
print('/', count, ' birth years found/')


# check if each matched sentence belong to different person
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Year_of_Birth']) != 0:
        print(df_sents.loc[i, 'Name'], '->', df_sents.loc[i, 'Year_of_Birth'], '\n')


# write rules to find death places
def find_death_place(text):
    death_places = []

    # create a spacy doc
    doc = nlp(text)

    # define the pattern
    pattern = [{'SHAPE': 'dddd'},
               {'LOWER': 'in'},
               {'POS': 'PROPN'},
               {'ORTH': ')'}]

    pattern1 = [{'LOWER': 'died'},
                {'LOWER': 'in'},
                {'POS': 'PROPN'}]

    # match class object
    matcher = Matcher(nlp.vocab, validate=True)
    matcher.add("death_place", None, pattern, pattern1)

    matches = matcher(doc)

    # find patterns in the text
    for i in range(0, len(matches)):
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        death_places.append(str(token))

    return death_places


# apply function
df_sents['Place_of_Death'] = df_sents['Sent'].apply(find_death_place)

# count the matched sentences
count=0
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Place_of_Death']) != 0:
        count += 1

print('\n')
print('/', count, ' death places found/')


# check if each matched sentence belong to different person
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Place_of_Death']) != 0:
        print(df_sents.loc[i, 'Name'], '->', df_sents.loc[i, 'Place_of_Death'], '\n')


# write rules to find death years
def find_death_year(text):
    death_years = []

    # create a spacy doc
    doc = nlp(text)

    # define the pattern
    pattern = [{'TEXT': '–'},
               {'SHAPE': 'd'},
               {'POS': 'PROPN'},
               {'SHAPE': 'dddd'},
               {'ORTH': ')'}]

    pattern1 = [{'ORTH': '–'},
                {'SHAPE': 'dd', 'OP': '?'},
                {'POS': 'PROPN'},
                {'SHAPE': 'dddd'},
                {'ORTH': ')'}]

    pattern2 = [{'ORTH': '–'},
                {'ORTH': 'died', 'OP': '?'},
                {'SHAPE': 'd'},
                {'POS': 'PROPN'},
                {'SHAPE': 'dddd'},
                {'ORTH': 'in'}]

    pattern3 = [{'ORTH': '–'},
                {'ORTH': 'died', 'OP': '?'},
                {'SHAPE': 'dd'},
                {'POS': 'PROPN'},
                {'SHAPE': 'dddd'},
                {'ORTH': 'in'}]

    # match class object
    matcher = Matcher(nlp.vocab, validate=True)
    matcher.add("death_year", None, pattern, pattern1, pattern2, pattern3)

    matches = matcher(doc)

    # find patterns in the text
    for i in range(0, len(matches)):
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        death_years.append(str(token))

    return death_years


# apply function
df_sents['Year_of_Death'] = df_sents['Sent'].apply(find_death_year)

# count the matched sentences
count=0
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Year_of_Death']) != 0:
        count += 1

print('\n')
print('/', count, ' death years found/')


# check if each matched sentence belong to different person
for i in range(len(df_sents)):
    if len(df_sents.loc[i, 'Year_of_Death']) != 0:
        print(df_sents.loc[i, 'Name'], '->', df_sents.loc[i, 'Year_of_Death'], '\n')


# clean dataframe by deleting rows not matching any patterns
for i in range(len(df_sents)):
    birth_place = df_sents.loc[i, 'Place_of_Birth']
    birth_year = df_sents.loc[i, 'Year_of_Birth']
    death_place = df_sents.loc[i, 'Place_of_Death']
    death_year = df_sents.loc[i, 'Year_of_Birth']

    if birth_place == birth_year == death_place == death_year:
        df_sents.drop(index=i, inplace=True)

# reset index and remove all sentences
df_sents.reset_index(inplace=True)
df_sents.drop(['index', 'Sent'], axis=1, inplace=True)

print('\n')
print('/Result Summary/\n', df_sents.head())
print(df.shape, '\n')

# convert dataframe to a csv file
df_sents.to_csv('WikiBioCorpus.csv')
