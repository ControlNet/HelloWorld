# Initialize variables to judge whether the number is valid.
# Integer 0 means missing and integer 1 means collected.
fandomCondition = 0
hobbiesCondition = 0
sportsCondition = 0
nerdCondition = 0

# Let users choose which number they want to input.
# Use infinity loop to provide enough chances to users to choose options.
while True:
    # Let users select an option to choose which number they want to input.
    usersOption = input("Please select one option:" + "\n" + "A: Input Fandom Score." + "\n" + \
                        "B: Input Hobbies Score." + "\n" + "C: Input The number of sports. " + "\n" + \
                        "D: Calculate Nerd Score" + "\n" + "E: Print Nerd Rating of Students" + "\n" + "F: Exit")

    if usersOption.lower() == "a":
        # Input Fandom Score from users.
        fandomScore = input("Please input your Fandom Score. ")
        # Check Fandom Score availability.
        while fandomScore.isdigit() is False or int(fandomScore) <= 0:
            if fandomScore == "":
                fandomScore = input("Error! Fandom Score is missing. Please input your Fandom Score again. ")
            elif fandomScore.isdigit() is False:
                fandomScore = input("Error! Fandom Score should be positive integer. Please input your Fandom Score. ")
            elif int(fandomScore) <= 0:
                fandomScore = input("Error! Fandom Score should be positive. Please input your Fandom Score. ")
            else:
                fandomScore = input("Error! Please input your Fandom Score again. ")
        # Mark that the Fandom Score has been input.
        fandomCondition = 1
        # Print the result of Fandom Score.
        print("Your Fandom Score is", fandomScore)

    elif usersOption.lower() == "b":
        # Input Hobbies Score from users.
        hobbiesScore = input("Please input your Hobbies Score. ")
        # Check Hobbies Score availability.
        while hobbiesScore.isdigit() is False or int(hobbiesScore) % 4 != 0:
            if hobbiesScore == "":
                hobbiesScore = input("Error! Hobbies Score is missing. Please input your Hobbies Score again. ")
            elif hobbiesScore.isdigit() is False:
                hobbiesScore = input("Error! Hobbies Score should be non-negative integer. Please input again. ")
            elif int(hobbiesScore) % 4 != 0:
                hobbiesScore = input("Error! Hobbies Score should be multiples of 4. Please input Hobbies Score. ")
            else:
                hobbiesScore = input("Error! Please input your Hobbies Score again. ")
        # Mark that the Hobbies Score has been input.
        hobbiesCondition = 1
        # Print the result of Hobbies Score.
        print("Your Hobbies Score is", hobbiesScore)

    elif usersOption.lower() == "c":
        # Input the Number of Sports played from users.
        sportsNum = input("Please input the number of sports played. ")
        # Check the Number of Sports availability.
        while sportsNum.isdigit() is False or int(sportsNum) < 0:
            if sportsNum == "":
                sportsNum = input("Error! Hobbies Score is missing. Please input your number of sports played again. ")
            elif sportsNum.isdigit() is False:
                sportsNum = input("Error! The number should be non-negative integer. Please input again. ")
            elif int(sportsNum) < 0:
                sportsNum = input("Error! The number should be positive. Please input the number of sports again. ")
            else:
                sportsNum = input("Error! Please input your number of sports again. ")
        # Mark that the number of sports played has been input.
        sportsCondition = 1
        # Print result of the number of sports played.
        print("Your number of sports played is", sportsNum)

    elif usersOption.lower() == "d":
        if fandomCondition == 1 and hobbiesCondition == 1 and sportsCondition == 1:
            # Calculate the Nerd Score and print.
            print("The Nerd Score is XXX.")
            # Mark the Nerd Score has been calculated.
            nerdCondition = 1
        else:
            # If the data is not completely collected, print error message.
            # Show which score is not collected.
            if fandomCondition == 0:
                print("Fandom Score missing.")
            if hobbiesCondition == 0:
                print("Hobbies Score missing.")
            if sportsCondition == 0:
                print("The number of sports missing.")
            print("Please input missing scores, and then re-calculate the Nerd Score.")

    elif usersOption.lower() == "e":
        # Calculate the Nerd Rating and print.
        if nerdCondition == 1:
            print("The Nerd Rating is XXX.")
        else:
            print("Please calculate the Nerd Score first, and then print the Nerd Rating.")

    elif usersOption.lower() == "f":
        # Exit the loop, also means the program.
        break

    else:
        # If user does not enter a valid option, print error message.
        print("Please input a correct character to choose an option.")
