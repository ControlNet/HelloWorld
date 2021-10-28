def countStudentClass(studentScore_list):
    if len(studentScore_list) < 1:
        print("Please add at least 1 item into the list")
        return 0

    nerdCount_list = [0] * 7  # initialize the output list

    # Initialize the condition marker to judge if the values in the list are string or integers.
    integerCondition = True
    stringCondition = True
    for student in range(len(studentScore_list)):
        '''
        Check if the contents of list are all string or integer. Only two situations are possible in the task3. One 
        happens when the inputs are all string, which means the input are imported from menu.py. The program need 
        convert it to be calculated. Another situation happens when the inputs are all provided as integers; the program
        can calculate directly. If other situations happen, there will be an error message printed.
        '''
        integerCondition = integerCondition and type(studentScore_list[student]) == int
        stringCondition = stringCondition and type(studentScore_list[student]) == str

    while True:
        # If all contents in the list are integers, all of them will be calculated.
        if integerCondition is True and stringCondition is False:
            for student in range(len(studentScore_list)):
                if 0 <= studentScore_list[student] < 1:
                    nerdCount_list[0] += 1  # Mark as Nerdlite.
                elif 1 <= studentScore_list[student] < 10:
                    nerdCount_list[1] += 1  # Mark as Nerdling.
                elif 10 <= studentScore_list[student] < 100:
                    nerdCount_list[2] += 1  # Mark as Nerdlinger.
                elif 100 <= studentScore_list[student] < 500:
                    nerdCount_list[3] += 1  # Mark as Nerd.
                elif 500 <= studentScore_list[student] < 1000:
                    nerdCount_list[4] += 1  # Mark as Nerdington.
                elif 1000 <= studentScore_list[student] < 2000:
                    nerdCount_list[5] += 1  # Mark as Nerdrometa.
                elif studentScore_list[student] >= 2000:
                    nerdCount_list[6] += 1  # Mark as Nerd Supreme.
            break

        # If all contents in the list are string, all of them will be transfer to integers and enter the loop again.
        elif stringCondition is True and integerCondition is False:
            for student in range(len(studentScore_list)):
                studentScore_list[student] = int(studentScore_list[student])
            integerCondition = True
            stringCondition = False

        # If any other situations happens, which is mentioned above, the program will print an error message.
        else:
            print("The numbers input don't fit the requirement. Please check it again.")
            break

    return nerdCount_list


if __name__ == '__main__':

    # test cases
    # studentScore_list = []  #
    studentScore_list = [23, 76, 1300, 600]  # output should be [0, 0, 2, 0, 1, 1, 0]

    try:
        print(countStudentClass(studentScore_list))

    except e:
        print(e)
        raise
