import pandas as pd
# import nltk
# nltk.download('punkt')
from nltk import word_tokenize


df = pd.read_csv('all_annotated.tsv', sep='\t')
# onlt select English comments
df_text =  df.loc[  df['Definitely English']==1,['Tweet']]

# Lowercase the tweet text
df_text['Tweet'] = df_text['Tweet'].str.lower()

# Remove Extra Whitespaces
def remove_whitespace(text):
    return  " ".join(text.split())

df_text['Tweet'] = df_text['Tweet'].apply(remove_whitespace)

# Removal Tags
import re
def remove_tag(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
df_text['Tweet'] = df_text['Tweet'].apply(remove_tag)

# Removal of URLs
url_pattern = re.compile(r'https?://\S+|www\.\S+')
def remove_urls(text):
    return url_pattern.sub(r'', text)

df_text['Tweet'] = df_text['Tweet'].apply(remove_urls)

# Tokenization
df_text['Tweet'] = df_text['Tweet'].apply(lambda X: word_tokenize(X))

# Spelling Correction
from spellchecker import SpellChecker   # python -m pip install pyspellchecker
spell = SpellChecker()
def spell_check(text):
    result = []
    
    for word in text:
        correct_word = spell.correction(word)
        if not (correct_word is None):
            result.append(correct_word)
    
    return result
df_text['Tweet'] = df_text['Tweet'].apply(spell_check)


# Stopword
import nltk
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

def remove_stopwords(text):
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)
            
    return result
df_text['Tweet'] = df_text['Tweet'].apply(remove_stopwords)

# Removing Punctuations
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")

def remove_punct(text):
    
    lst=tokenizer.tokenize(' '.join(text))
    return lst

df_text['Tweet'] = df_text['Tweet'].apply(remove_punct)

# Removing Frequent Words
from nltk import FreqDist
def frequent_words(df):
    
    lst=[]
    for text in df.values:
        lst+=text[0]
    fdist=FreqDist(lst)
    return fdist.most_common(10)

freq_words = frequent_words(df_text)
lst = []
for a,b in freq_words:
    lst.append(a)

def remove_freq_words(text):
    
    result=[]
    for item in text:
        if item not in lst:
            result.append(item)
    
    return result
    
df_text['Tweet']=df_text['Tweet'].apply(remove_freq_words)

# # Lemmatization
# from nltk.stem import WordNetLemmatizer
# from nltk import word_tokenize,pos_tag

# def lemmatization(text):
    
#     result=[]
#     wordnet = WordNetLemmatizer()
#     for token,tag in pos_tag(text):
#         pos=tag[0].lower()
        
#         if pos not in ['a', 'r', 'n', 'v']:
#             pos='n'
            
#         result.append(wordnet.lemmatize(token,pos))
    
#     return result
# df_text['Tweet']=df_text['Tweet'].apply(lemmatization)

# Stemming
from nltk.stem import PorterStemmer

def stemming(text):
    porter = PorterStemmer()
    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result
df_text['Tweet']=df_text['Tweet'].apply(stemming)

df_text.to_csv('processed_data_final.csv', index=False, sep="\t")