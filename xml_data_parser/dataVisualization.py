"""
Introduction of this program in dataVisualization.py
This program is to visualize the data gathered in previous process.
Two figures will be generated:
wordNumberDistribution.png describes the distribution of vocabulary size in this xml input file;
postNumberTrend.png describes the trend of post number of questions and answers separately.
"""

import parser as ps  # import from the file of task 2
import re
import matplotlib.pyplot as plt


def visualizeWordDistribution(inputFile, outputImage):
    # this function is to visualize the distribution of vocabulary size
    indexList = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "others"]
    sizeList = parseFile(inputFile)[0]  # Count the Vocabulary size for each line
    wordDistributionFigure = createBarChart(sizeList, indexList)  # create a bar chart for it
    wordDistributionFigure.show()
    wordDistributionFigure.savefig(outputImage)  # save this bar chart with the determined file name


def visualizePostNumberTrend(inputFile, outputImage):
    # this function is to visualize the post number trend of questions and answers
    questionAmountDict, answerAmountDict = parseFile(inputFile)[1:3]
    postNumberTrendFigure = createLineChart(questionAmountDict, answerAmountDict)  # create a line chart for it
    postNumberTrendFigure.show()
    postNumberTrendFigure.savefig(outputImage)  # save this line chart with the determined file name


def parseFile(inputFile):
    # this function is to count the vocabulary size in each line in this xml file
    # initialize the list to present the amount of 0-10, 10-20,..., 90-100, others (left inclusive)
    vocabularySizeList = [0] * 11
    # initialize the dictionary to present the amount of questions and answers separately for different quarters
    questionAmountDict = {}
    answerAmountDict = {}
    with open(inputFile, "r", encoding="utf-8") as inputHandle:
        for line in inputHandle:
            if re.search("<\?xml version=.+? encoding=.+?\?>", line) != None:  # check if it is the first line in xml
                continue
            elif re.search("<.?posts>", line) != None:  # check if it is the second or last line in xml
                continue
            else:  # Other lines which are wanted body lines
                lineParser = ps.Parser(line)  # instantiate the Parser class for each line
                # manipulate the list for vocabulary size
                vocabularySizeList = countVocabularySize(lineParser, vocabularySizeList)
                # manipulate the dictionaries for post amount in quarters
                postAmountTuple = countPostAmountInQuarters(lineParser, questionAmountDict, answerAmountDict)
    return vocabularySizeList, postAmountTuple[0], postAmountTuple[1]


def countVocabularySize(parser, sequence):
    # this function is to count vocabulary size and record in the list
    size = parser.vocabularySize  # get vocabulary size of this line
    index = size // 10  # locate the index of the vocabularySizeList for this line
    if index > 10:
        index = 10
    sequence[index] += 1  # count this line in the list
    return sequence


def countPostAmountInQuarters(parser, qDict, aDict):
    # this function is to count the post amount of questions and answers in each quarter
    quarter = parser.getDateQuarter()
    postType = parser.getPostType()
    # consider when this line belongs to question or answer
    if postType == "Question":
        qDict = manipulatePostAmountDict(qDict, quarter)
    elif postType == "Answer":
        aDict = manipulatePostAmountDict(aDict, quarter)
    return qDict, aDict


def manipulatePostAmountDict(dictionary, quarter):
    # this function is to manipulate the post amount dictionary to count the amount for each line
    if quarter not in dictionary:  # if the quarter is not existed before, create it
        dictionary[quarter] = 1
    else:  # else, plus one to count the amount
        dictionary[quarter] += 1
    return dictionary


def createBarChart(valueList, indexList):
    # this function is to create a bar chart for distribution of vocabulary size
    figure = plt.figure(0, [8, 4.8])
    plt.bar(indexList, valueList, 0.5)  # plot the bar chart with width is 0.5
    plt.ylim(0, 2000)  # set the limitation of y-axis
    # set labels in x-axis and y-axis
    plt.ylabel("Post Amount")
    plt.xlabel("Vocabulary Size")
    plt.title("The Distribution of Vocabulary Size")  # set title of this graph
    return figure


def createLineChart(qDict, aDict):
    # this function is to create a line chart for post amount trend
    xAxis = qDict.keys()  # extract the x-axis items from the dictionary
    qAmount = qDict.values()  # determine the y-axis values for questions
    aAmount = aDict.values()  # determine the y-axis values for answers
    figure = plt.figure(1, [10, 4])
    # plot 2 lines for questions and answers
    plt.plot(xAxis, qAmount, "orange", label="Question")
    plt.plot(xAxis, aAmount, "blue", label="Answer")
    plt.ylim(0, 400)  # set the limitation of y-axis
    # set labels in x-axis and y-axis
    plt.ylabel("Post Amount")
    plt.xlabel("Quarter")
    plt.title("The Trend of Post Amount")  # set title of this graph
    plt.legend(loc="upper right")  # print the legend at upper right of this graph
    return figure


if __name__ == "__main__":
    f_data = "data.xml"
    f_wordDistribution = "wordNumberDistribution.png"
    f_postTrend = "postNumberTrend.png"

    visualizeWordDistribution(f_data, f_wordDistribution)
    visualizePostNumberTrend(f_data, f_postTrend)
