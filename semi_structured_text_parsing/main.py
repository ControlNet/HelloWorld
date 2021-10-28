#!/usr/bin/env python
# coding: utf-8

# # Semi-Structured Text Data Parsing
# 
# Version: 4.0
# 
# Environment: Python 3.7 and Jupyter notebook
# 
# Libraries used:
# * pandas (for dataframe, included in Anaconda Python 3.7) 
# * re (for regular expression, included in Anaconda Python 3.7) 
# * json (for save json file, included in Anaconda Python 3.7)

# ## 1. Introduction
# This project is aim to extract data from semi-structured text files. There are 150 patents in the file named `data.txt`. <b>The following tasks are what need to be finished:</b>
# 
# 
# - Extract grant_id, patent_kind, patent_title, number_of_claims, citations_examiner_count, citations_applicant_count,inventors, claims_text and abstrct from the `data.txt` 
# - Then transform all these data into a csv file and json file.
# 
# More details for the task will be given in the following sections.

# ## 2.  Import libraries 

# In[1]:


import pandas as pd
import re
import json


# ## 3. Examining and loading data
# At first step, the file `data.txt` will be explored and print 10 lines below.

# In[2]:


with open('data.txt','r') as input_file:
    for i in range(10):
        print(input_file.readline())


# According to the first XML document, there is a XML declaration which is `<?xml...?>`. In addition, `<us-patent-grant...>` is a root tag which can help us to identify each patent. Based on these information, extracing every XML individually is feasible.
# 
# The number of the XML declaration is able to ascertain how many patents are contained in the file. Therefore, the regex for `<?xml...?>` is defined. `?`, `version=` and `"?` are exist in each xml declaraton, but there a special meaning for `?`. The `\` is necessary to allow the `?` to be used without special meaning.
# 
# ```python
# regex = r'<\?xml version=".+?" encoding=".+?"\?>'
# ```
# `version` and `encoding=` are followed by any characters, so the non-greedy pattern `.+?` is applied here (Python, 2019).
# 
# Here the built-in function `re.findall()` is used to find all matched contents as a list returned (Python, 2019). 

# In[3]:


with open('data.txt','r') as input_file:
    text = input_file.read()
regex = r'<\?xml version=".+?" encoding=".+?"\?>'
print(len(re.findall(regex, text)))


# It seems the txt file is merged by 150 xml files with xml format. Therefore, one of the xml part should be printed to examine. 
# 
# Every XML document has only one XML declaration. The whole strings is suitable to be divided into 150 parts by the declaration. Here, re.split is used to divide all contents and the result is a list of strings(patents) and using filter() to avoide existing an empty elements in the list. 
# 

# In[4]:


xmls = re.split(regex, text)
xmls = list(filter(lambda x:x!='', xmls)) # filter the element which is not blank
print(len(xmls))


# The list `xmls` is a list of strins with length 150. In this case, the first one is chosen to inspect.
# 

# In[5]:


xml_example = xmls[0]
print(xml_example[:1000])


# ## 4. XML parser design
# 
# The data of belows should be extracted:
# * grant_id: a unique ID for a patent grant consisting of alphanumeric characters. 
# * patent_kind: a category to which the patent grant belongs. 
# * patent_title: a title given by the inventor to the patent claim.
# * number_of_claims:  an integer denoting the number of claims for a given grant. 
# * citations_examiner_count: an integer denoting the number of citations made by the examiner for a given patent grant (0 if None)
# * citations_applicant_count: an integer denoting the number of citations made by the applicant for a given patent grant (0 if None)
# * inventors: a list of the patent inventors’ names ([NA] if the value is Null).
# * claims_text: a list of claim texts for the different patent claims ([NA] if the value is Null).
# * abstract: the patent abstract text (‘NA’ if the value is Null)
# 
# Every parsed data contains above information, therefore creating a list for every data is necessary. The method to_attribute_list is created within the class.

# For building the json file needed, the json_dict is set for it. As for more details, see 5.3 please.
# ```python
#     json_dict = {}
# ```

# A class for store these data is created.

# In[6]:


class Data:
    json_dict = {}
    
    def __init__(self, grant_id, patent_title, kind, number_of_claims, inventors, citations_applicant_count,
                 citations_examiner_count, claims_text, abstract):
        # constructor
        self.grant_id = grant_id
        self.patent_title = patent_title
        self.kind = kind
        self.number_of_claims = number_of_claims
        self.inventors = inventors
        self.citations_applicant_count = citations_applicant_count
        self.citations_examiner_count = citations_examiner_count
        self.claims_text = claims_text
        self.abstract = abstract
        Data.json_dict[self.grant_id] = self.to_json_dict()

    @staticmethod
    def build_data_from_xml(xml):
        # build object of Data class from given xml format text
        # these functions used will be defined below
        grant_id = extract_grant_id(xml)  # see 4.1
        patent_title = extract_patent_title(xml)  # see 4.2
        kind = extract_kind(xml)  # see 4.3
        number_of_claims = extract_number_of_claims(xml)  # see 4.4
        inventors = extract_inventors(xml)  # see 4.5
        citations_applicant_count, citations_examiner_count = extract_citations_count(xml)  # see 4.6
        claims_text = extract_claims_text(xml)  # see 4.7
        abstract = extract_abstract(xml)  # see 4.8
        return Data(grant_id, patent_title, kind, number_of_claims, inventors, citations_applicant_count,
                    citations_examiner_count, claims_text, abstract)

    def to_attribute_list(self):
        # generate the elements for each Data object
        attribute_list = [self.grant_id,
                          self.patent_title,
                          self.kind,
                          self.number_of_claims,
                          self.inventors,
                          self.citations_applicant_count,
                          self.citations_examiner_count,
                          self.claims_text,
                          self.abstract]
        return attribute_list
    
    def to_json_dict(self):
        # generate the value of json_dict for each Data object
        return {
            "patent_title": self.patent_title,
            "kind": self.kind,
            "number_of_claims": int(self.number_of_claims),
            "investors": self.inventors,
            "citations_applicant_count": int(self.citations_applicant_count),
            "citations_examiner_count": int(self.citations_examiner_count),
            "claims_text": self.claims_text,
            "abstract": self.abstract
        }


# ### 4.1 Extract grant_id
# Comparing between `sample_input.txt` and `sample_output.csv`, the grant_id is in the file tag for each XML files. For instance:
# 
# `<us-patent-grant lang="EN" dtd-version="v4.5 2014-04-03" file="US10357261-20190723.XML" status="PRODUCTION" id="us-patent-grant" country="US" date-produced="20190709" date-publ="20190723">`
# 
# ```python
#     regex = r'<us-patent-grant.+?file="(.+?)".+?>'
# ```
# 
# The ID is in the double quote which is following `file=`, grouping is used here for extracting the ID. The information after `file="US10357261-20190723.XML` can be any character, so non-greedy pattern `.+?` is used here. Besides, the ID information is alphanumeric characters. 
# 
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).
# 
# After getting the value from by regular expression, next step is to extract the grant id. According to the `asmaple_inpt.txt` and `sample_output.csv` the ID is the alphanumeric characters before `-` ("US10357261-20190723.XML").
# 
# In `file="US10357261-20190723.XML"`, the `US10357261` is the grant_id.
# 
# Using `-` to split the sting within the group#1 is feasible the get the grant_id which is in index 0 of grant_id list. (Python, 2019)

# In[7]:


def extract_grant_id(xml):
    regex = r'<us-patent-grant.+?file="(.+?)".+?>'
    value = re.search(regex, xml)[1]
    grant_id = value.split("-")[0]
    return grant_id

grant_id = extract_grant_id(xml_example)
print(grant_id)


# ### 4.2 Extract patent_title
# Using the same method, the patent titie is in the "invention_title" tag. For instance:
# 
# `<invention-title id="d2e53">Single-use orthopedic surgical instrument</invention-title>`
# 
# The text in the middle is the content need to be extracted. So, the regular expression is defined below.
# 
# ```python
#     regex = r'<invention-title id=".+?">(.+?)<\/invention-title>'
# ```
# 
# `<invention-title id="` + any character  + `">` + any character + `</invention-title>` should be the structure of regex. The second any character would be the patent_title, so grouping the  contents is necessary, and there is a `/` in `</invention-title>`, therefore `/` should be applied here to making the `/` without special meaning. (Python, 2019)
# 
# Here the built-in function `re.search()` is also used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).

# In[8]:


def extract_patent_title(xml):
    regex = r'<invention-title id=".+?">(.+?)<\/invention-title>'
    patent_title = re.search(regex, xml)[1]
    return patent_title

patent_title = extract_patent_title(xml_example)
print(patent_title)


# ### 4.3 Extract patent_type
# 

# 
# The type of patent is stored inside the first "kind" tag, which are presented as a code, such as "A" and "B1". For this situation, it should be transfored to more detailed value like the output sample. According to USPTO Kind Codes(2019), the kind codes are "A1","B1","B2" or "S1" etc. Hence, creating a dictionary to transfer the code into specific informtaion is a suitable method.
# 
# Using the same method, the patent kind code is in the "kind" tag. For instance:
# 
# `<kind>B2</kind>`
# 
# Therefore, using the similar pattern like 4.2 is the suitable way. The regex is showed below.
# ```python
#     regex = r'<kind>(.*?)</kind>'
# ```
# 
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).
# 
# The code is implemented below.

# In[9]:


def extract_kind(xml):
    regex = r'<kind>(.*?)</kind>'
    kind_dict = {"A": "Utility Patent Application published on or after January 2, 2001.",
                 "A1": "Utility Patent Application published on or after January 2, 2001.",
                 "B1": "Utility Patent Grant (no pre-grant publication) issued on or after January 2, 2001.",
                 "B2": "Utility Patent Grant (with pre-grant publication) issued on or after January 2, 2001.",
                 "S": "Design Patent",
                 "S1": "Design Patent"
                 }
    kind_code = re.search(regex, xml)[1]
    kind = kind_dict[kind_code] # transfer codes to the kind information which is needed
    return kind

kind = extract_kind(xml_example)
print(kind)


# ### 4.4 Extract number_of_claims
# For number of claims, there is also a specific tag named "number-of-claims" for it. For instance:
# 
# `<number-of-claims>18</number-of-claims>`
# 
# Obviously, 18 is the target. The method is same as 4.2.
# 
# Therefore, using the similar pattern like 4.2 is the suitable way. The regex is showed below.
# ```python
#     regex = r'<kind>(.*?)</kind>'
# ```
# 
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).

# In[10]:


def extract_number_of_claims(xml):
    regex = r'<number-of-claims>(.+?)<\/number-of-claims>'
    number_of_claims = re.search(regex, xml)[1]
    return number_of_claims  # the return value is str

number_of_claims = extract_number_of_claims(xml_example)
print(number_of_claims)


# ### 4.5 Extract inventors
# The attribute inventors means a collection of several inventors. All inventors data are located in "inventors" tag, and for each inventor there is a "inventor" tag related to it. So, in the "inventors" tag, there will be one or several "inventor" contents included.
# 
# For each "inventor" tag, the first name and last name, which are needed to be extracted, is in the tag "first-name" and "last-name" seperately.
# For example:
# `<last-name>Kugler</last-name><first-name>Andrew</first-name>`
# 
# The regular expression is showed below:
# 
# 1. extract the all contents between "inventors" tags: grouping the contents within `<inventors>` and `</inventors>`. The contents can use non-greedy pattern `.+?` to match. 
# ```python
#    regex = r'<inventors>(.+?)<\/inventors>'
# ```
# 2. The exctracted contents would contain one or many inventors name, so using similar method to extract the names. re.findall can get a result which is a list contains all of inventors name. Each inventors name will be stored in a tuple. (Python, 2019)
# 
# ```python
#    all_name = re.findall(r'<last-name>(.+?)</last-name><first-name>(.+?)</first-name>',inventors_value)
# ```
# Finally, the final result should be like Python list, but without quotation mark. For example, `[Andrew Kugler,John A. Williams, II]`.

# In[11]:


def extract_inventors(xml):
    xml_in_one_line = xml.replace("\n", "")
    regex = r'<inventors>(.+?)<\/inventors>'
    # extract the large "inventors" tag
    inventors_value = re.search(regex, xml_in_one_line)[1]
    # if there is no inventor, mark it as "[NA]"
    if inventors_value == None:
        return '[NA]'
    
    all_name = re.findall(r'<last-name>(.+?)</last-name><first-name>(.+?)</first-name>',inventors_value)
    # create a list to store all inventors' names
    name_list = []
    for each in all_name:
        each_name = list(each)[::-1]
        new_name = ' '.join(each_name)
        name_list.append(new_name)
    
    inventors = "[" + ",".join(name_list) + "]"
    return inventors

extract_inventors(xml_example)


# ### 4.6 Extract citations counts
# In the xml text, the "us-references-cited" tag contains all cites values. In each cite part is handled by a "us-citation" tag. In each "us-citation" tag, the type of citation (cited by applicant or examiner) is stored in "category" tag.
# 
# Therefore, the method is clear. Firstly, get the value of "us-reference-cited" tag, then, count each type of citation and export it.
# 
# The regular expression to get the value of this tag is displayed below.
# ```python
#     regex = r'<us-references-cited>(.+?)<\/us-references-cited>'
# ```
# 
# 1. extract the information within "us-references-cited" tag (using group to extract all contents)
# 2. use str.count to caculate the citation counts
# 
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).

# In[12]:


def extract_citations_count(xml):
    regex = r'<us-references-cited>(.+?)<\/us-references-cited>'
    xml_in_one_line = xml.replace("\n", "")
    minor_xml = re.search(regex, xml_in_one_line)[1]
    applicant_count = minor_xml.count("cited by applicant")
    examiner_count = minor_xml.count("cited by examiner")
    return str(applicant_count), str(examiner_count)  # the return values are str

extract_citations_count(xml_example)


# ### 4.7 Extract claims
# From the example, it is clearly matched the claims text in output file with the value in "claims" tag in xml. The method to extract that is to filter the value of "claims" tag in xml file, then delete all tags, leaving the pure text.
# 
# The regular expression to get the value of this tag is displayed below.
# ```python
#     regex = r'<claims id="claims">(.+?)<\/claims>'
# ```
# 
# 1. The method to extract text within "claims" tag is smilar to above section
# 2. remove all tags in text: use `<[^>]*>` to match all tags. There are many `>` within text, so we need match every tag. There should not be a `>` in a tag, so `[^>]` is used here and it may appear 0 or more time,because 0 or more characters are here. (Python, 2019)
# 
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).

# In[13]:


def extract_claims_text(xml):
    regex = r'<claims id="claims">(.+?)<\/claims>'
    xml_in_one_line = xml.replace("\n", "")
    minor_xml = re.search(regex, xml_in_one_line)[1]
    
    if minor_xml == None:
        return 'NA'
    
    claims_text = re.sub(r"<[^>]*>", "", minor_xml)
    return '['+claims_text+']'

extract_claims_text(xml_example)


# ### 4.8 Extract abstract
# According to the example provided, all abstract texts are inside of "abstract" tag, and there are several tags like `<p id="p-0001" num="0000">` and `</p>`inside so these need to be deleted. The method is same as 4.7.
# ```python
#     regex = r'<abstract id="abstract">(.+?)</abstract>'
# ```
# Here the built-in function `re.search()` is used for search the matched sub-string and use index 1 to get the value of it (Python, 2019).
# 
# Then what need to do is getting the value of it, cleanning it to a pure text.

# In[14]:


def extract_abstract(xml):
    regex = r'<abstract id="abstract">(.+?)</abstract>'
    xml_in_one_line = xml.replace("\n", "")
    abstract_search = re.search(regex, xml_in_one_line)
    if abstract_search is None:
        return 'NA'
    minor_xml = abstract_search[1]
    abstract = re.sub(r"<[^>]*>", "", minor_xml)
    return abstract

extract_abstract(xml_example)


# ## 5. Output
# ### 5.1 Implement parser
# After collecting data from each xml logs, next step is to implement these functions to all datas inside the input file.
# 

# ```python
# class Data:
#     json_dict = {}
#     
#     def __init__(self, grant_id, patent_title, kind, number_of_claims, inventors, citations_applicant_count,
#                  citations_examiner_count, claims_text, abstract):
#         # constructor
#         self.grant_id = grant_id
#         self.patent_title = patent_title
#         self.kind = kind
#         self.number_of_claims = number_of_claims
#         self.inventors = inventors
#         self.citations_applicant_count = citations_applicant_count
#         self.citations_examiner_count = citations_examiner_count
#         self.claims_text = claims_text
#         self.abstract = abstract
#         Data.json_dict[self.grant_id] = self.to_json_dict()
# 
#     @staticmethod
#     def build_data_from_xml(xml):
#         # build object of Data class from given xml format text
#         # these functions used will be defined below
#         grant_id = extract_grant_id(xml)  # see 4.1
#         patent_title = extract_patent_title(xml)  # see 4.2
#         kind = extract_kind(xml)  # see 4.3
#         number_of_claims = extract_number_of_claims(xml)  # see 4.4
#         inventors = extract_inventors(xml)  # see 4.5
#         citations_applicant_count, citations_examiner_count = extract_citations_count(xml)  # see 4.6
#         claims_text = extract_claims_text(xml)  # see 4.7
#         abstract = extract_abstract(xml)  # see 4.8
#         return Data(grant_id, patent_title, kind, number_of_claims, inventors, citations_applicant_count,
#                     citations_examiner_count, claims_text, abstract)
# ```
# By using the `build_data_from_xml()` method and `map()` function, all xml texts in list `xmls` will be extracted and an object of Data class will be built for each xml text, and then store in the variable `data_list`.
# 
# For each object of class Data, there is a one-to-one match between its attributes and the values of each columns for each row. In other words, the element in `data_list` contains everything needed for output.

# In[15]:


# transfer all xml format texts to Data object
data_list = list(map(Data.build_data_from_xml, xmls))
data_list[:10]


# ### 5.2 Output csv files
# <b>Here the method `to_attribute_list(self)` in Data class is used. The details are below:</b>
# 
# ```python
# def to_attribute_list(self):
#     # generate the elements for each Data object
#     attribute_list = [self.grant_id,
#                       self.patent_title,
#                       self.kind,
#                       self.number_of_claims,
#                       self.inventors,
#                       self.citations_applicant_count,
#                       self.citations_examiner_count,
#                       self.claims_text,
#                       self.abstract]
#     return attribute_list
# ```   
# This method is to make each csv rows by inserting the attributes which are collected in section 4.

# First of all, creating a DataFrame is necessary to be prepared before outputing csv format by using Pandas.
# 
# Including the data in a list structure is a feasible way to creat a dataframe, and using a row  oriented approach by using pandas `from_records` is a approach to create a dataframe(Moffitt, 2019).

# In[16]:


data_matrix = list(map(Data.to_attribute_list,data_list))  # 2-dimension list structure
columns_name = "grant_id,patent_title,kind,number_of_claims,inventors,citations_applicant_count,citations_examiner_count,claims_text,abstract".split(",")


# In[17]:


df = pd.DataFrame.from_records(data=data_matrix,columns=columns_name)
df.head()


# Then, the dataframe created need exported in to a `.csv` files without index, and save as `data.csv`. Using the to_csv( ) [function](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html#pandas.DataFrame.to_csv) is a approprite way to export the file (Pandas, 2019).

# In[18]:


df.to_csv('data.csv', index=None)


# ### 5.3 Output json file

# According to the sample output json file, the structure of json file is clearly showed below:
# 
# ```json
# {"US10357643":
#      {"patent_title":"...",
#       "kind":"...",
#       "number_of_claims":20,
#       "inventors":"[name1,name2,...]",
#       "citations_applicant_count":8,
#       "citations_examiner_count":2,
#       "claims_text":"...",
#       "abstract":"..."
#       },
# "US10362643":
#      {"patent_title":"...",
#       "kind":"...",
#       "number_of_claims":16,
#       "inventors":"[name1,name2,...]",
#       "citations_applicant_count":0,
#       "citations_examiner_count":14,
#       "claims_text":"...",
#       "abstract":"..."
#       },
#  ...
#  }
# ```

# The json output file has 2 nested levels.
# 1. grant id
# 2. patent title, kind, number of claims, inventors, citations applicant count, citations examiner count, claims_text and abstract
# 
# For the first level, the key is `grant_id` and value is a collection of key-value pairs which contains other attributes;
# 
# For the second level, the keys are the name of attributes, and the values are the value related to each attribute.
# 
# To write the output json file, dictionary data structure is chosen. To implement that, a class variable `json_dict` is created.
# 
# <b> Please note the code below is only the important parts of original one </b>
#     
# ```python
# class Data:
#     json_dict = {}
#     ...
# ```
# 
# As known in previous, for each object, there is a key-value pair related. The key of it is the value of` self.grant_id`. The value of that is the dictionary which are the pairs of the attributes and their values. To implement it, the method `to_json_dict(self)` is created. It returns the value of `json_dict` for each object.
# 
# ```python
#     def to_json_dict(self):
#         # generate the value of json_dict for each Data object
#         return {
#             "patent_title": self.patent_title,
#             "kind": self.kind,
#             "number_of_claims": int(self.number_of_claims),
#             "investors": self.inventors,
#             "citations_applicant_count": int(self.citations_applicant_count),
#             "citations_examiner_count": int(self.citations_examiner_count),
#             "claims_text": self.claims_text,
#             "abstract": self.abstract
#         }
# ```
# 
# As it is needed to insert the key-value pair into `json_dict` for each object. The most convinent way is to do this step inside the `__init__` method. Here is the code below, please see the last line.
# 
# ```python
#     def __init__(self, grant_id, patent_title, kind, number_of_claims, inventors, citations_applicant_count,
#                  citations_examiner_count, claims_text, abstract):
#         # constructor
#         self.grant_id = grant_id
#         self.patent_title = patent_title
#         self.kind = kind
#         self.number_of_claims = number_of_claims
#         self.inventors = inventors
#         self.citations_applicant_count = citations_applicant_count
#         self.citations_examiner_count = citations_examiner_count
#         self.claims_text = claims_text
#         self.abstract = abstract
#         Data.json_dict[self.grant_id] = self.to_json_dict()
# ```
# 
# After the step to create all objects (see 5.1), the class variable `json_dict` is also built as well. Next step is to output it. However, the default way that Python transfering the dictionary to string will use single quotations rather than double one (json needs double quotations). So there is a step to replace single quotations to double by using `str.replace()` method. After everything done, the final step is to write the `data.json` file as output.

# In[19]:


with open('data.json', 'w') as output_json:
    output_json.write(json.dumps(Data.json_dict))


# ## 6. Summary

# This project is aim to measure the basic techniques of handling the text file by Python.
# The outcome achieved after using these basic techniques were:
# 
# - <b>XML parsing and data extraction</b>: Using basic re package knowledge to prase the data and extracing  required data.
# - <b>Creating data frame</b>: By using `pandas` package and built-in function `DataFrame.from_records()` to to create a dataframe.
# - <b>Outputing data into `.csv` file </b>: Built-in function `DataFrame.to_csv()` was applied to generate a `.csv` file.
# - <b>Outputing data into `.json` file </b>: Build a dictionary and then generate a `.json` file by using built-in function `file.write()`.

# ## 7. References

# - Moffitt, C. (2019). Creating Pandas DataFrames from Lists and Dictionaries - Practical Business Python. Retrieved from https://pbpython.com/pandas-list-dict.html
# - Pandas. (2019). pandas.DataFrame.to_csv — pandas 0.25.1 documentation. Retrieved from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
# - USPTO Kind Codes. (2019). Retrieved from https://www.uspto.gov/patents-application-process/patent-search/authority-files/uspto-kind-codes#s3
# - Python. (2019). re — Regular expression operations — Python 3.7.4 documentation. Retrieved from https://docs.python.org/3/library/re.html
