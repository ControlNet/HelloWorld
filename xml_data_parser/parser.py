"""
Introduction of this program in parser.py
This program is used for collecting required data for further analysis by building a class. This class can collect row
ID, post type, date, clean body and vocabulary size for one line.
"""

import preprocessData as pd  # import from preprocessData which is the file for task 1
import re


class Parser:
    """
    The Parser class is for collect required data from inputString. Other program can easily get the data from this
    class. Once the instance is set, the required data will be generated as well, which contains row ID, post type,
    date, clean body and vocabulary size.
    """

    def __init__(self, inputString):
        self.inputString = inputString
        self.ID = self.getID()
        self.type = self.getPostType()
        self.dateQuarter = self.getDateQuarter()
        self.cleanBody = self.getCleanedBody()
        self.vocabularySize = self.getVocabularySize()

    def __str__(self):
        # print ID, Question/Answer/Others, creation date quarter, the main content, and vocabulary size
        text = "ID: " + self.ID + "\n" + "Type: " + self.type + "\n" + "Creation Date: " + self.dateQuarter + "\n" + \
               "The main content: " + self.cleanBody + "\n" + "Vocabulary size: " + str(self.vocabularySize)
        return text

    def getID(self):
        # return the row Id of this line
        # the type of rowId is string
        item = re.search('\d+(?=" PostTypeId)', self.inputString)  # search the row id in the string
        if item is not None:
            index = item.span()  # identify the location of row id in the string
            rowId = self.inputString[index[0]:index[1]]  # get the row id
            return rowId

    def getPostType(self):
        # return question, answer or other
        # the type of postTypeId is string
        item = re.search('\d+(?=" CreationDate)', self.inputString)  # search the post type id in the string
        if item is not None:
            index = item.span()  # identify the location of post type id in the string
            postTypeId = self.inputString[index[0]:index[1]]  # get the post type id
            # transform the post type id into actual post type (question, answer or other)
            if postTypeId == "1":
                postType = "Question"
            elif postTypeId == "2":
                postType = "Answer"
            else:
                postType = "Other"
            return postType

    def getDateQuarter(self):
        # return the quarter and year of this line
        # the type of result is string
        item = re.search('(?<=CreationDate=").+(?=" Body)', self.inputString)  # search the creation date
        if item is not None:
            index = item.span()  # identify the location of creation date in the string
            dateStr = self.inputString[index[0]:index[1]]  # get the string contains date and time
            year = dateStr[0:4]  # get the year data from the date string
            month = dateStr[5:7]  # get the month data from the date string
            # transform the month to quarter
            if int(month) in (1, 2, 3):
                quarter = "Q1"
            elif int(month) in (4, 5, 6):
                quarter = "Q2"
            elif int(month) in (7, 8, 9):
                quarter = "Q3"
            elif int(month) in (10, 11, 12):
                quarter = "Q4"
            result = year + quarter  # combine the year and quarter as the result
            return result

    def getCleanedBody(self):
        # return the body of this line
        outputStr = pd.preprocessLine(self.inputString)  # process the line from xml form to clean body
        return outputStr

    def getVocabularySize(self):
        # return the number of unique words in the body of this line
        # the type of return is integer
        wordsSet = set()  # initialize a set to store unique words
        wordsStr = self.cleanBody.lower()  # using the cleaned body of this line
        wordsStr = wordsStr.replace("/n", " ")  # replace the linefeed to space for easy count
        wordsStr = re.sub('\)?\W +\W?| \W+', " ", wordsStr)  # delete all punctuations and spaces
        wordsList = wordsStr.split(" ")  # using list to record every word
        for each in wordsList:
            wordsSet.add(each)  # transfer the item from list to set to delete repeating item
        if "" in wordsSet:
            wordsSet.remove("")
        return len(wordsSet)
