#!/usr/bin/env python
# coding: utf-8

# # PDF Text Feature Extraction
# 
# Environment: Python 3.7 and Jupyter notebook
# 
# Libraries used:
# * requests (for downloading pdf files from internet, included in Python 3.7)
# * os (for os operation, included in Python 3.7)
# * PyPDF2 (for extract text from pdf, downloaded from pip)
# * tabula (for extract table from pdf, downloaded from pip)
# * pandas (for data frame to manipulate data, included in Anaconda 3)
# * re (for extracting string, included in Python 3.7)
# * nltk (for English processing, downloaded from pip)
# * pdfminer (for extract text from pdf, downloaded from pip)
# * io (for using in function to extract pdf text, included in Python 3.7)
# * functools (for using `reduce()` to process list, included in Python 3.7)
# * multiprocessing (for using `Pool` to boost the speed of processing in MacOS and Linux, included in Python 3.7)
# * types (for using function annotation for better cooperation and readability, incluede in Python 3.7)
# * platform (for recognizing the operating system of user's, included in Python 3.7)
# * tqdm(for visualize the processing bar, downloaded from pip)

# ## 1. Introduction

# This project is aim to :
# 
# Generate a sparse representation for Paper Bodies (i.e. paper text without Title, Authors,
# Abstract and References). The sparse representation consists of two files:
# 
# 1. Vocabulary index file
# 2. Sparse count vectors file
# 
# Generate a CSV file (stats.csv) containing three columns:
# 1. Top 10 most frequent terms appearing in all Titles
# 2. Top 10 most frequent Authors
# 3. Top 10 most frequent terms appearing in all Abstracts

# ## 2.  Import libraries 

# In[1]:


get_ipython().system('pip install pypdf2')
get_ipython().system('pip install tabula-py')
get_ipython().system('pip install pdfminer')
get_ipython().system('pip install nltk')
get_ipython().system('pip install pdfminer.six')


# In[1]:


import requests
import os
import PyPDF2
import tabula
import pandas as pd
import re
import nltk
import nltk.data
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer 
from nltk.tokenize import MWETokenizer
from nltk.probability import *
# pdfminer below is using for parse pdf file to text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from functools import reduce
from multiprocessing import Pool
from types import FunctionType
from tqdm.notebook import tqdm
import platform


# In[20]:


nltk.download('punkt')


# ## 3. Download PDF

# <b>Tabula is used here for extract table from PDF `(data.pdf)` into DataFrame.</b> (Ariga, 2019)

# In[2]:


# parse the pdf table to a dataframe
urls = tabula.read_pdf("data.pdf", output_format="dataframe", pages="all")[0]
urls


# <b>All pages have a title which affect the process to download PDF files, so the filename title shuld be deleted here.<b/>

# In[3]:


# modify the data frame as preparation of auto-downloading
index_for_drop = urls[urls.filename=="filename"].index
urls = urls    .drop(index_for_drop)    .reset_index()    .drop(["index"],axis=1)


# <b>Downloading 200 pdf files</b>
# 
# Please be patient, this chunk will spend some minutes to download 200 pdf files.

# In[6]:


def download_files_from_df(df):
    # iterate rows in data frame
    for row in df.values:
        name, url = row
        # skip if the file is already downloaded
        if os.path.exists(name):
            continue
        # download the pdf content
        try:
            req = requests.get(url)
        except:
            continue
        # save as a file
        with open("pdf/" + name, "wb") as pdf_file:
            pdf_file.write(req.content)

download_files_from_df(urls)
print("Download successful.")


# ## 3. Converting PDF to Text
# 
# <b>Define the function to extract pdf content into text.</b> (Mike, 2019)

# In[4]:


def extract_pdf_content(pdf_name):
    pdf_rm = PDFResourceManager()
    io = StringIO()
    converter = TextConverter(rsrcmgr=pdf_rm, outfp=io, codec='utf-8', laparams=LAParams())
    with open(pdf_name, 'rb') as file:
        interpreter = PDFPageInterpreter(pdf_rm, converter)
        page_nos = set()
        for page in PDFPage.get_pages(file, page_nos, maxpages=0, password="", caching=True,
                                      check_extractable=True):
            interpreter.process_page(page)
    output_str = io.getvalue()
    converter.close()
    io.close()
    return output_str


# <b>Using a pdf file as an example to examine the content</b>

# In[5]:


pdf_example = extract_pdf_content("pdf/PP7133.pdf")
pdf_rows_example = pdf_example.split("\n")
pdf_rows_example[:10]


# ### 3.1 Creating functions to get Paper Bodies

# ```python
# #1 This function is used to delete all hex_codes
# def delete_hex_code(row):
#     new_row = row.replace("\x0c", "")
#     return new_row
# ```
# =======================================================================
# ```python
# #2 In this function #3 function will be called to get all content of paper body
# def extract_body(row_list: list) -> list:
#     body_begin_index = find_index(r"1 Paper Body", row_list)+1
#     body_end_index = find_index(r"2 References", row_list)
#     return row_list[body_begin_index:body_end_index]
# 
# #3 identify the index of start and end of paper body.
# def find_index(regex, a_list):
#     for index in range(len(a_list)):
#         if re.search(regex, a_list[index]) is not None:
#             return index
# ```

# <b>According to above example, there are some hex_code in the content. Following function will delete all hex_codes.</b>

# In[6]:


"""
implement the function to delete hex code "\x0c", 
which is used for split the page in pdfminer parser
"""
def delete_hex_code(row):
    new_row = row.replace("\x0c", "")
    return new_row

# split into rows
pdf_rows_example = pdf_example.split("\n")
# drop blank rows
pdf_rows_example = list(filter(lambda x: x != '', pdf_rows_example))
# apply the function to delete hex code
pdf_rows_example = list(map(delete_hex_code, pdf_rows_example))
pdf_rows_example[:10]


# Each Paper body begin with `1 Paper Bosy` and end with `2 References`, so these two will be used as the regex in the following function.
# 
# More detail about the relationship between following 2 functions please refer to the explaination at the beginning of 3.1

# In[7]:


# define the function to extract body part from row_list
def extract_body(row_list: list) -> list:
    body_begin_index = find_index(r"1 Paper Body", row_list)+1
    body_end_index = find_index(r"2 References", row_list)
    return row_list[body_begin_index:body_end_index]

# define the function to get index from given regular expression pattern
def find_index(regex, a_list):
    for index in range(len(a_list)):
        if re.search(regex, a_list[index]) is not None:
            return index


# In[8]:


# implement the function
pdf_body_rows_example = extract_body(pdf_rows_example)
pdf_body_rows_example[:5]


# ### 3.2 Merge all bodies in pdf files together in a nested list.

# The #2 function will call the #1 function!
# #1 function is used to make a list for pdf contents without blank rows and hex_code
# 
# #2 function is used to extract the body.
# 
# ```python
# if is_ok_for_pool:
#     p = Pool()
#     # apply the function to get body_rows from given file name
#     body_row_list = list(p.map(filename_to_body_rows, filename_list))
# else:
#     body_row_list = list(map(filename_to_body_rows, filename_list))
# ```
# Using #2 function with map() to get paper body and to get a `2-dimensions list` . There are 200 list in this list. 
# 
# As Windows cannot using `Pool` to boost the process in iPython/Jupyter, Windows machine runs single-thread in this process 

# In[9]:


#1 this function is used to processing from given pdf file name to row_list
def filename_to_rows(filename: str) -> list:
    # extract pdf content from given file name
    try:
        string = extract_pdf_content("pdf/" + filename)
    except:
        return []
    # process pdf content to row_list
    row_list = string.split("\n")
    row_list = list(filter(lambda x: x != '', row_list))
    row_list = list(map(delete_hex_code, row_list))
    return row_list

#2 this function is used to process from given file name, and extract body part
def filename_to_body_rows(filename: str) -> list:
    row_list = filename_to_rows(filename)
    body_row = extract_body(row_list)
    return body_row

# collect the file name from the data frame defined in previous
filename_list = [each for each in urls.filename]
# determine if the operating system is compatible to the multithreading.
# as Windows cannot perform Pool inside the Jupyter Notebook
is_ok_for_pool = platform.system() in ("Darwin", "Linux")

# if the operating system is MacOS or Linux, using multithreading to boost the process
if is_ok_for_pool:
    p = Pool()
    # apply the function to get body_rows from given file name
    body_row_list = list(p.map(filename_to_body_rows, filename_list))
else:
    body_row_list = list(map(filename_to_body_rows, tqdm(filename_list)))

# filter the blank rows
body_row_list = list(filter(lambda x: x != [], body_row_list))


# ## 3.3 Convert bodies in pdf files together in one list.

# The following function is used to make list flat from nested list. (Python, 2019)

# In[10]:


# define a function to make list flat from nested list
def to_flat_list(list_1: list, list_2: list) -> list:
    return list_1 + list_2

# using reduce() to flat list
merged_body_rows = reduce(to_flat_list, body_row_list)
len(merged_body_rows)


# ## 4. Sparse Feature Generation

# The following operations are necessary
# * A. The word tokenization must use the following regular expression, r"[A-Za-z]\w+(?:[-'?]\w+)?"
# * B. The context-independent and context-dependent (with the threshold set to %95) stop words must be removed from the vocab. The context-independent stop words list (i.e, stopwords_en.txt) provided in the zip file must be used.
# * C. Unigram tokens should be stemmed using the Porter stemmer. (be careful that stemming performs lower casing by default)
# * D. Rare tokens (with the threshold set to 3%) must be removed from the vocab.
# * E. Tokens must be normalized to lowercase except the capital tokens appearing in the middle of a sentence/line. (use sentence segmentation to achieve this)
# * F. Tokens with the length less than 3 should be removed from the vocab.
# * G. First 200 meaningful bigrams (i.e., collocations), based on highest total frequency in the corpus, must be extracted and included in your tokenization process. Bigrams should not include context-independent stopwords as part of them and they should be separated using double underscore i.e. “__” (example: “artifical__intelligence”)
# 
# The order of above operations will be ordered as <b>`EAGBDCF`</b>.

# ### 4.1 Operation E: lowercase words except the capital tokens appearing in the middle of a sentence.
# 

# Now, as a first step, we need to lowercase the beginning words of each sentence. we should divide the contents into sentences. The NLTK data package includes a pre-trained Punkt tokenizer for English ,and this tokenizer divides a text into a list of sentences(nltk, 2019).

# In[11]:


# define the function to transfer the first character of a sentence into lowercase
#1
def to_lowercase_for_first_word(sentence: str) -> str:
    try:
        new_sentence = sentence[0].lower()+sentence[1:]
    except:
        new_sentence = sentence
    return new_sentence

# there are some words using "-" connecting in 2 different rows,
# defining the function to merge this type words together
#2
def join_splited_words(string: str) -> str:
    split_pattern = r'(?<=\w)- (?=\w+?)'
    new_string = re.sub(split_pattern, "", string)
    return new_string

# define the function to segment sentence, and process by calling functions above
#3
def to_lowercase_as_sentence(body_rows: list) -> list:
    sentence_spliter = nltk.data.load('tokenizers/punkt/english.pickle')
    body_string = " ".join(body_rows).strip()
    body_string = join_splited_words(body_string)
    sentence_list = sentence_spliter.tokenize(body_string)
    sentence_list_lower = list(map(to_lowercase_for_first_word, sentence_list))
    return sentence_list_lower

# apply the function
merged_sentences = to_lowercase_as_sentence(merged_body_rows)
merged_sentences[:3]


# ### 4.2 Operation A: The word tokenization must use the following regular expression, r"[A-Za-z]\w+(?:['?]\w+)?" 

# Then, using `nltk.tokenize.RegexpTokenizer` to tokenize words from sentence list with the regular expression provided. (nltk, 2019)

# In[12]:


# define the function to tokenize word 
def tokenize_word(sentences: list, regex: str) -> list:
    tokenizer = RegexpTokenizer(regex)
    tokens = tokenizer.tokenize(" ".join(sentences))
    return tokens

# apply the function with the regular expression provided
merged_word_tokens = tokenize_word(merged_sentences, r"[A-Za-z]\w+(?:[-'?]\w+)?")
print(len(merged_word_tokens))
unique_words = list(set(merged_word_tokens))
unique_words.sort()
print(len(unique_words))
unique_words[:10]


# ### 4.3 Operation G: First 200 meaningful bigrams  (i.e., collocations), based on highest total frequency in the corpus, must be extracted and included in your tokenization process. Bigrams should not include context-independent stopwords as part of them and they should be separated using double underscore i.e. “\_\_”  (example: “artifical__intelligence”) 

# Then, using `nltk.collocations.BigramAssocMeasures` and `nltk.collocations.BigramCollocationFinder` to find 200 most frequency bigrams from given tokens.

# In[13]:


# define the function to filter the bigrams without stop words
def filter_bigrams_meaningful(bigrams):
    is_not_included_stopwords = lambda x: not (x[0].lower() in stopwords or x[1].lower() in stopwords)
    return list(filter(is_not_included_stopwords, bigrams))

# define the function to generate top200 bigrams
def generate_top200_bigrams(bigrams):
    # recursion
    # drop bigrams with stop words
    bigrams = filter_bigrams_meaningful(bigrams)
    length = len(bigrams)
    return bigrams[:200]

# get stop words set from "stopwords_en.txt" which is provided
stopwords = set([line.rstrip('\n') for line in open("stopwords_en.txt")]) 
# create the objects needed to implement bigram parser
bigram_measures = nltk.collocations.BigramAssocMeasures()
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(merged_word_tokens)
bigram_finder.apply_freq_filter(2)
# apply the function in previous
top_bigrams = generate_top200_bigrams(bigram_finder.nbest(bigram_measures.raw_freq, int(len(merged_word_tokens)/4)))
print(len(top_bigrams))
top_bigrams[:5]


# After getting the top 200 bigrams, next step is to re-token bigrams in tokens by implementing `nltk.tokenize.MWETokenizer`

# In[14]:


# define the function to replace the bigrams in the token list
def replace_bigrams(tokens: list, bigrams: list) -> list:
    mwetokenizer = MWETokenizer(bigrams, separator='__')
    new_tokens = mwetokenizer.tokenize(tokens)
    return new_tokens

# apply the function
merged_word_token_processed = replace_bigrams(merged_word_tokens, top_bigrams)
len(merged_word_token_processed)


# ### 4.4 Operation B: The context-independent and context-dependent (with the threshold set to 95%) stop words must be removed from the vocab. The context-independent stop words list (i.e, stopwords_en.txt) provided in the zip file must be used. 

# First step is to remove context independent words, which is provided in `stopwords_en.txt`

# In[15]:


# define the function to remove tokens which is stop words
def remove_context_independent_words(tokens: list) -> list:
    new_tokens = list(filter(lambda x: x.lower() not in stopwords, tokens))
    return new_tokens

# drop the tokens which is duplicated, and apply the function
word_tokens_processed = remove_context_independent_words(merged_word_token_processed)
print(len(word_tokens_processed))
word_tokens_processed[:5]


# To remove the context dependent words, it requires a statistic information for each word about the number of documents that each word is contained by them. Therefore, to find it, next step is to process the tokens for each pdf files. 
# 
# Please be patient. The code chunk below need few seconds to run.

# In[16]:


# define the function to apply the processing in previous for each pdf file
def generate_word_token_from_body_row(body_row: list) -> list:
    # process the body rows to sentences
    sentences = to_lowercase_as_sentence(body_row)
    # process the sentences to word tokens
    tokens = tokenize_word(sentences, r"[A-Za-z]\w+(?:[-'?]\w+)?")
    # process the tokens with replacing bigrams
    tokens = replace_bigrams(tokens, top_bigrams)
    # process the tokens to drop stop words
    tokens = remove_context_independent_words(tokens)
    return tokens

# apply the function
word_tokens_each_pdf = list(map(generate_word_token_from_body_row, body_row_list))
# the list copied is used for count vector
word_tokens_each_pdf_list = word_tokens_each_pdf.copy()  
# drop the dulicated elements
word_tokens_each_pdf = list(map(set, word_tokens_each_pdf))
#word_tokens_each_pdf


# Then, after getting the tokens for each pdf files, define the function `merge_set()` and use `reduce()` to merge all set for each tokens. Meanwhile, this function `merge_set()` has a side effect, which can count the number of files that contains the word, and store the infomation in the dictionary `word_dict`.
# 
# As for `word_dict`, the keys of the dictionary contains all words that in 200 files, and the values of the dictionary contains the number of pdf files containing the word.

# In[17]:


# initialize the dictionary to store the information about the frequency
word_dict = {}
# set the count of words with in first pdf as 1
for word in word_tokens_each_pdf[0]:
    word_dict[word] = 1
# define the function to merge token sets from each pdf,
# and using the side effect to get the frequency information
def merge_set(word_set1: set, word_set2: set):
    # if the word appears in previous, increment the count by 1
    inter_set = word_set2.intersection(word_set1)
    for word in inter_set:
        word_dict[word] += 1
    # if the word does not appear in previous, set the count to 1
    difference_set = word_set2.difference(word_set1)
    for word in difference_set:
        word_dict[word] = 1
    # return the merged set
    return word_set2.union(word_set1)
    
# apply the function by reduce()
word_tokens_union = reduce(merge_set, word_tokens_each_pdf)


# ### 4.5 Operation D: Rare tokens (with the threshold set to 3%) must be removed from the vocab.
# 
# After getting the information in the previous step, using the keys in `word_dict` for each word to judge if it should be deleted.

# In[18]:


# define the function to delete countext dependent words and rare words
def remove_context_dependent_words_and_rare_words(tokens: set) -> set:
    delete_list = []
    pdf_amount = len(filename_list)
    for key, value in word_dict.items():
        if value >= pdf_amount*0.95 or value < pdf_amount*0.03:
            delete_list.append(key)
    new_tokens = set(filter(lambda x: x not in delete_list, tokens))
    return new_tokens

# apply the function
word_tokens_union_processed = remove_context_dependent_words_and_rare_words(word_tokens_union)
len(word_tokens_union_processed)


# ### 4.6 Operation C: Unigram tokens should be stemmed using the Porter stemmer. 
# Then, using `nltk.stem.PorterStemmer` to stem the word tokens.

# In[19]:


# define the to_stem function to process the words which is lowercase
def to_stem(word: str) -> str:
    if word[0].isupper():
        new_word = word
    elif word.count("_") > 0:
        new_word = word
    else:
        new_word = stemmer.stem(word)
    return new_word

# initialize the PorterStemmer, and apply the function
stemmer = PorterStemmer()
stemmer.mode = PorterStemmer.NLTK_EXTENSIONS
final_tokens = list(map(to_stem, word_tokens_union_processed))
# delete the duplicate tokens and sort
final_tokens = list(set(final_tokens))
final_tokens.sort()
# display
print(len(final_tokens))
final_tokens[:5]


# ### 4.7 Operation F: Tokens with the length less than 3 should be removed from the vocab.

# Then, next step is to delete short words whose length is less than 3, by defining the `delete_short_words` function and applying.

# In[20]:


# define the function to delete the short words with the length < 3
def delete_short_words(tokens: list) -> list:
    is_not_short = lambda x: len(x) >= 3
    new_tokens = list(filter(is_not_short, tokens))
    return new_tokens

# apply the function, and delete duplicated tokens
final_tokens_fixed = list(delete_short_words(final_tokens))
print(len(final_tokens_fixed))
final_tokens_fixed[:5]


# ### 4.8 Output vocabulary index file

# After collecting each necessary data from the bodies of 200 pdf files, next step is to output the files required.
# 
# To output vocabulary index file, firstly combine the index number to each words, which are sorted by alphabetical ascending order.

# In[21]:


# process the tokens, and assign each words with indexes
final_tokens_fixed.sort()
vocabulary_index_pairs = list(zip(final_tokens_fixed, list(range(len(final_tokens_fixed)))))
vocabulary_index_pairs[:5]


# For each word, generate a string representing a row, and use `reduce()` to process all words in `vocabulary_index_pairs`.

# In[22]:


# define the function to generate each row in output file,
# and merge them together by reduce() function
def generate_vocabulary_index(formal_text: str, pair: tuple) -> str:
    row = pair[0] + ":" + str(pair[1])
    return formal_text + "\n" + row

# apply the function with reduce()
vocabulary_index_text = reduce(generate_vocabulary_index, [""]+vocabulary_index_pairs).lstrip("\n")
vocabulary_index_text[:100]


# At last, write the file by using `file.write()` method.

# In[23]:


# write the file as output
with open("vocab.txt", "w", encoding="UTF-8") as output_file:
    output_file.write(vocabulary_index_text)


# ### 4.9 Generate sparse count vectors file

# To generate sparse count vectors file, for each word, the number of pdf files containing the word should be collected. Although, the information has been collected in 4.4 Operation B, in this section, the word tokens should be stemmed. Therefore, the stem processed should be applied in `word_tokens_each_pdf_list` which is copied from the `word_tokens_each_pdf` in 4.4 Operation B.
# 
# So, define a high-order function `apply_func()` to generate a function for map the list.

# In[24]:


# define a function to generate function for map inside the list
def apply_func(func: FunctionType) -> FunctionType:
    def map_list(x):
        return list(map(func, x))
    return map_list

# apply the function
apply_to_stem = apply_func(to_stem)
stemmed_tokens_each_pdf_list = list(map(apply_to_stem, tqdm(word_tokens_each_pdf_list)))
# stemmed_tokens_each_pdf_list


# Comparing the words in the keys of `vocabulary_index_pairs` with the word tokens for each pdf files will get the necessary data of sparse count vectors. Then, store the data for a pdf file for each words as strings that are required, and store the count vectors in a list for each pdf files.

# In[25]:


# initialize the list to store count vector
count_vectors = []
pdf_names = list(map(lambda x: x.replace(".pdf", ""), urls.filename))

# iterate each pdf tokens
for pdf_index in tqdm(range(len(stemmed_tokens_each_pdf_list))):
    # initialize the count_vector string with the file name
    count_vector = pdf_names[pdf_index] + ","
    pdf_tokens = stemmed_tokens_each_pdf_list[pdf_index]
    
    # iterate each token to append the string
    for word, word_index in vocabulary_index_pairs:
        count = pdf_tokens.count(word)
        if count == 0:
            # if this word does not exist in the document, skip
            continue
        else:
            # append the string
            count_vector += (str(word_index)+":"+str(count)+",")
    # delete the last useless ","
    count_vector = count_vector.rstrip(",")
    count_vectors.append(count_vector)


# Join the count vector list together with `\n` to ensure the vectors for each pdf files locating in different rows to merge the string.

# In[26]:


# join the count vectors for each pdf into a string
count_vector_text = "\n".join(count_vectors)
count_vector_text[:100]


# The last step is to output the file.

# In[27]:


# save the output file
with open("count_vectors.txt", "w", encoding="UTF-8") as output_file:
    output_file.write(count_vector_text)


# ## 5.  Statistics Generation

# ### 5.1 Merge all Titles, Authors and Abstracts into 3 lists
# #### 5.1.1 Function for extracting title

# In[28]:


# define the function to extract title from row_list
def extract_title(row_list: list) -> str:
    end_index = find_index(r"Authored by:", row_list)
    title_rows = row_list[:end_index]
    title_string = " ".join(title_rows)
    return title_string

extract_title(pdf_rows_example)


# #### 5.1.2  Function for extracting authors

# In[29]:


# define the function to extract authors from row_list
def extract_authors(row_list: list) -> list:
    start_index = find_index(r"Authored by:", row_list) + 1
    end_index = find_index(r"Abstract", row_list)
    author_list = row_list[start_index: end_index]
    abstract_text_total = "".join(author_list)
    return author_list

extract_authors(pdf_rows_example)


# #### 5.1.3  Function for extracting abstract

# In[30]:


# define the function to extract abstract from row_list,
# and process to lowercase which is needed
def processing_abstract(abstract_rows: list) -> str:
    sentence = to_lowercase_as_sentence(abstract_rows)
    return sentence

def extract_abstract(row_list: list) -> str:
    start_index = find_index(r"Abstract", row_list) + 1
    end_index = find_index(r"1 Paper Body", row_list)
    abstract_rows = row_list[start_index: end_index]
    sentence = processing_abstract(abstract_rows)
    abstract_text_total = " ".join(sentence)
    return abstract_text_total

extract_abstract(pdf_rows_example)[:100]


# #### 5.1.4 Get pdf contents and implement the functions (get 3 lists)
# This chunk will require several minutes to process 200 pdf files. Please be patient.

# In[33]:


# define the function which generate the rows of title, author and abstract seperately
def filename_to_extract_rows(filename: str) -> list:
    try:
        string = extract_pdf_content("pdf/" + filename)
    except:
        return []
    row_list = string.split("\n")
    row_list = list(filter(lambda x: x != '', row_list))
    title_row = extract_title(row_list)
    author_row = extract_authors(row_list)
    abstract_row = extract_abstract(row_list)
    return (title_row, author_row, abstract_row)

# apply the function
# if the operating system is MacOS or Linux, using multithreading to boost the process
if is_ok_for_pool:
    p2= Pool()
    row_lists = list(p2.map(filename_to_extract_rows, filename_list))
else:
    row_lists = list(map(filename_to_extract_rows, tqdm(filename_list)))

# store each part to each variables for future usage
title_row_list = [each[0] for each in row_lists]
author_row_list = [each[1] for each in row_lists]
abstract_row_list = [each[2] for each in row_lists]
# delete the blank rows in title
title_row_list = list(filter(lambda x: x != [], title_row_list))


# ### 5.1.5  Function for lowercasing titles

# In[34]:


# define the function to transfer the title to lowercase
def to_lowercase_title(title_rows: list) -> list:
    title_string_list = list(map(lambda x: x.lower(), title_rows))
    return title_string_list

# apply the function
merged_lower_title_list = to_lowercase_title(title_row_list)
merged_lower_title_list[:10]


# ### 5.1.6 Title words toknization 
# Using tokenize_word( ) function which is defined in 4.2 to get the tokoens
# ```python
# def tokenize_word(sentences: list, regex: str) -> list:
#     tokenizer = RegexpTokenizer(regex)
#     tokens = tokenizer.tokenize(" ".join(sentences))
#     return tokens
# ```

# In[35]:


# get word tokens
title_tokens = tokenize_word(merged_lower_title_list, r"[A-Za-z]\w+(?:[-'?]\w+)?")
print(len(title_tokens))
print(title_tokens[:10])


# ### remove stop words in title
# In order to remove context-independent stop words the remove_context_independent_words( ) which was defined in <b>4.4</b> will be used here.
# ```python
# def remove_context_independent_words(tokens: list) -> list:
#     new_tokens = list(filter(lambda x: x.lower() not in stopwords, tokens))
#     return new_tokens
# ```

# In[36]:


# remove stop words
title_tokens_processed = remove_context_independent_words(title_tokens)
print(len(title_tokens_processed))
print(title_tokens_processed[:10])


# ### Get tokens frequency information
# Counting the frequency of each token by using FreqDist( ) which is from the nltk(nltk.tokenize package — NLTK 3.4.5 documentation", 2019). This will record the number of time each token has occured.
# 
# It is hard to get the content which is processed by the FreqDist( ). Therefore, we use most_common cuntion to convert it into a list.

# In[37]:


# get the frequency distribution for each words
title_top10_freq = FreqDist(title_tokens_processed)
title_top10_freq = title_top10_freq.most_common()
print(title_top10_freq[:15])


# In case of ties in any of the above fields, settle the tie based on alphabetical ascending order.
# The following functions can help to generate the top 10 terms which is ordered firstly by the frequency descending order then ordered by alphabetical ascending order.
# ```python
# pairs = sort_2d_list_by_2keys(pairs,1,True,0,False)
# ```
# Firstly,  #3 function order the index1 by descending then order the index0 by alphabetical ascending order.

# In[38]:


# define the function to generate function to extract element in given indexes,
# for using in list.sort()
def get_element_extractor(*index_tuple: tuple) -> FunctionType:
    def extract_elements(a_list):
        new_list = []
        for index in index_tuple:
            new_list.append(a_list[index])
        return new_list
    return extract_elements

# define the function to sort a pair list by given 2 indexes, 
# and with if it is reversed or not for each index
def sort_2d_list_by_2keys(
    a_list: list, 
    index1: int, 
    reverse1: bool, 
    index2: int, 
    reverse2: bool
) -> list:
    new_list = a_list.copy()
    # sort by index1
    get_index1 = get_element_extractor(index1)
    new_list.sort(key=get_index1, reverse=reverse1)
    # extract unique index1 elements for second sort
    elements_in_index1 = list(set([x[index1] for x in new_list]))
    elements_in_index1.sort(reverse=reverse1)
    # sort by index2
    result_list = []
    for element in elements_in_index1:
        sub_list = list(filter(lambda x: x[index1] == element, new_list))
        get_index2 = get_element_extractor(index2)
        sub_list.sort(key=get_index2, reverse=reverse2)
        result_list += sub_list
    return result_list
       
# define the function to get the top 10 terms
def get_top10_terms(top10_freq):
    # determine the count in rank 10 term
    top10_threshold = top10_freq[9][1]
    # collect the terms that achieves the count larger or equals to the top 10
    pairs = list(filter(lambda x: x[1] >= top10_threshold, top10_freq))
    # sort the list by the count in descending, and with the alphabetical ascending order
    pairs = sort_2d_list_by_2keys(pairs,1,True,0,False)
    # get the top 10
    top10_pairs = pairs[:10]
    # get the words themselves
    top10_terms = list(map(lambda x: x[0], top10_pairs))
    return top10_terms

# apply the function
top10_title_terms = get_top10_terms(title_top10_freq)
top10_title_terms


# ### 5.1.7 Merge all pdfs' Authors into a nested list and get top 10 authors

# In[39]:


# get the author list for each pdf
author_row_list = list(filter(lambda x: x != [], author_row_list))
author_row_list[:3]


# In[40]:


# merge the author list by reduce()
merged_author_rows = reduce(to_flat_list, author_row_list)
print(merged_author_rows[:5])


# Counting the frequency of each author by using `FreqDist` which is from the nltk(NLTK, 2019). This will record the number of time each author has occured.
# 
# It is hard to get the content which is processed by the `FreqDist`. Therefore, we use most_common cuntion to convert it into a list.
# 
# Finally, using the same function we to get the top 10 most frequent authors:
# 
# ```python
# def get_top10_terms(top10_freq):
#     # determine the count in rank 10 term
#     top10_threshold = top10_freq[9][1]
#     # collect the terms that achieves the count larger or equals to the top 10
#     pairs = list(filter(lambda x: x[1] >= top10_threshold, top10_freq))
#     # sort the list by the count in descending, and with the alphabetical ascending order
#     pairs = sort_2d_list_by_2keys(pairs,1,True,0,False)
#     # get the top 10
#     top10_pairs = pairs[:10]
#     # get the words themselves
#     top10_terms = list(map(lambda x: x[0], top10_pairs))
#     return top10_terms
# ```
# 

# In[41]:


# get the top10 of author names by similar method
author_top10_freq = FreqDist(merged_author_rows)
author_top10_freq = author_top10_freq.most_common()
top10_authors = get_top10_terms(author_top10_freq)
print(top10_authors)


# ### 5.1.8  Merge all Abstracts in pdf files together and get top 10 tokens in abstract
# The method is same as above section.

# In[42]:


# get the abstract rows
abstract_row_list = list(filter(lambda x: x != [], abstract_row_list))
# abstract_row_list


# In[43]:


# get the tokens in abstract, and remove stop words
abstract_tokens = tokenize_word(abstract_row_list, r"[A-Za-z]\w+(?:[-'?]\w+)?")
abstract_tokens = remove_context_independent_words(abstract_tokens)
abstract_tokens[:10]


# In[44]:


# get top10 tokens in abstract
abstract_top10_freq = FreqDist(abstract_tokens).most_common()
top10_abstract_terms = get_top10_terms(abstract_top10_freq)
top10_abstract_terms


# ## 5.2 Generating CSV file
# ### 5.2.1 create a data frame to store the information about top 10 terms
# * First of all, creating a DataFrame is necessary to be prepared before outputing csv format by using Pandas.
# * From 5.1.6, 5.1.7 and 5.1.8, we have generate 3 lists which contain top 10 frequency terms.
# * Creating a dictionary for dataframe(Pandas, 2019).

# In[45]:


# create a data frame to store the information about top 10
df = pd.DataFrame({
    "top10_terms_in_abstracts": top10_abstract_terms,
    "top10_terms_in_titles": top10_title_terms,
    "top10_authors": top10_authors
})
df


# Then, the dataframe created need exported in to a `.csv` files without index, and save as `stats.csv`. Using the to_csv( ) [function](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv) is a approprite way to export the file (Pandas, 2019).

# In[46]:


# output to csv file
df.to_csv("stats.csv", index=None, encoding="UTF-8")


# To check the output file is readable by computer, using `pandas.read_csv()` to read the file which is output in previous to check its correction.

# In[47]:


# check if the csv file is readable for computer
pd.read_csv("stats.csv")


# ## References

# 1. Ariga, A. (2019). tabula-py: Extract table from PDF into Python DataFrame. Retrieved from https://blog.chezo.uno/tabula-py-extract-table-from-pdf-into-python-dataframe-6c7acfa5f302
# 2. Mike. (2019). Exporting Data from PDFs with Python - The Mouse Vs. The Python. Retrieved from http://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/#targetText=Extracting%20Text%20with%20PDFMiner,as%20father%20information%20about%20fonts
# 3. nltk. (2019). nltk.tokenize package — NLTK 3.4.5 documentation.  Retrieved from https://www.nltk.org/api/nltk.tokenize.html?highlight=english%20pickle
# 4. nltk. (2019). nltk.tokenize.regexp — NLTK 3.4.5 documentation. Retrieved from https://www.nltk.org/_modules/nltk/tokenize/regexp.html
# 5. Pandas. (2019). pandas.DataFrame — pandas 0.25.1 documentation. Retrieved from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# 6. Pandas. (2019). pandas.read_csv — pandas 0.25.1 documentation. Retrieved from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# 7. Python. (2019). 9.8. functools — Higher order functions and operations on callable objects — Python v3.1.5 documentation. Retrieved from https://docs.python.org/3.1/library/functools.html
