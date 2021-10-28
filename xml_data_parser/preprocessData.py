"""
Introduction of this program in preprocessData.py
This program is used for process data for QA websites, which is stored in a XML file formation, extracting question
data and answer data, then process it to a pure content and record into 2 separate txt files.
"""

import re


def preprocessLine(inputLine):
    # this function is used to preprocess the data in each line
    httpLine = replaceXMLChars(inputLine)  # replace XML special characters
    simpleLine = replaceHTTPTags(httpLine)  # delete HTTP tags
    bodyLine = filterBody(simpleLine)  # get the body part
    return bodyLine


def splitFile(inputFile, outputFile_question, outputFile_answer):
    # this function is used to preprocess the original file, and split them into two files.
    with open(inputFile, "r", encoding="utf-8") as inputHandle, open(outputFile_question, "w",
                    encoding="utf-8") as questionHandle, open(outputFile_answer, "w", encoding="utf-8") as answerHandle:
        for line in inputHandle:
            if '''PostTypeId="1"''' in line:
                # If the row is a question
                line = preprocessLine(line)  # let the line processed
                print("Processing question", line)  # print the process
                questionHandle.write(line + '\n')  # write this line into question.txt
            elif '''PostTypeId="2"''' in line:
                # If the row is a answer
                line = preprocessLine(line)  # let the line processed
                print("Processing answer", line)  # print the process
                answerHandle.write(line + '\n')  # write this line into answer.txt


def replaceXMLChars(string):
    # this function is used to replace the XML special characters into original forms
    replaceDict = {"&quot;": '"', "&apos;": "'", "&gt;": ">", "&lt;": "<", "&#xA;": " ", "&#xD;": " ",
                   "&mdash": "—", "&ndash;": "-", "&nbsp;": " ", "&euro;": "€", "&hellip;": "…", "&thinsp;": " ",
                   "&times;": "×", "&asymp;": "≈", "&div;": " ", "&micro;": "µ", "&frac12;": "½", "&pm;": "±"}
    # replace all "&amp" to "&", and considered repetitive situation especially for "&amp"
    while "&amp;" in string:
        string = string.replace("&amp;", "&")
    # replace other XML special characters
    for key, value in replaceDict.items():
        string = string.replace(key, value)  # collect all transform relations in a dictionary
    return string


def replaceHTTPTags(string):
    # this function is used to delete the HTTP tags
    string = string.replace("<", "", 1)  # delete the "<" in the beginning of the line
    string = string.replace(" />\n", "\n")  # delete the "/>" in the end of the line
    string = re.sub("<\/?.+?\/?>", "", string)  # delete all HTTP tags
    return string


def filterBody(string):
    # this function is used to extract the body content from the string
    string = re.sub('row Id="\d+" ', "", string)  # delete all row Id
    string = re.sub('PostTypeId="\d+" ', "", string)  # delete all PostTypeId
    string = re.sub('CreationDate=".+?" ', "", string)  # delete all CreationDate
    string = string.replace('Body="', "")  # delete Body= and a quotation mark
    string = string.replace(' "\n', "")  # delete the quotation mark and '\n' in the last
    string = string.lstrip(" ")  # delete all spaces in the last
    return string


if __name__ == "__main__":
    f_data = "data.xml"
    f_question = "question.txt"
    f_answer = "answer.txt"

    splitFile(f_data, f_question, f_answer)
