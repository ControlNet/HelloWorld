# Functionality: calculate the skill score by the equation
# x, y, z are inputs
def calculateSkillEquation(FandomScore, HobbiesScore, SportsNum):
    skillScore = 0  # initialize the output list

    # Check the validity of parameters for this function.
    # Check the type of parameters first.
    while True:
        if type(FandomScore) == int and type(HobbiesScore) == int and type(SportsNum) == int:
            # Check if the scores input should fit the requirement.
            if FandomScore > 0 and HobbiesScore >= 0 and HobbiesScore % 4 == 0 and SportsNum >= 0:
                # Calculate the Nerd Score by nerdScoreEquation function.
                skillScore = nerdScoreEquation(FandomScore, HobbiesScore, SportsNum)
                return skillScore
            else:
                # If not fit requirement, print an error message and quit the loop.
                print("The numbers input don't fit the requirement. Please check it again. ")
                break

        # Consider the situation that if the parameters are all import from menu.py, they are all string.
        elif type(FandomScore) == str and type(HobbiesScore) == str and type(SportsNum) == str:
            if FandomScore.isdigit and HobbiesScore.isdigit and SportsNum.isdigit:
                # Convert the type to integer.
                FandomScore = int(FandomScore)
                HobbiesScore = int(HobbiesScore)
                SportsNum = int(SportsNum)
                # Enter the loop again to check its validity.
                continue
            else:
                # These situation only happens if there is some issue in data delivery.
                # Print an error message.
                print("The numbers input don't fit the requirement. Please check it again. ")
                break
        else:
            # These situation only happens if there is some issue in data delivery like above one.
            # Print an error message.
            print("The numbers input don't fit the requirement. Please check it again. ")
            break


def nerdScoreEquation(x, y, z):
    result = x * ((42 * y ** 2) / (z + 1)) ** 0.5
    return result


if __name__ == '__main__':

    FandomScore, HobbiesScore, SportsNum = 1, 4, 1  # the output should be 18.33030277982336

    try:
        print(calculateSkillEquation(FandomScore, HobbiesScore, SportsNum))

    except e:
        print(e)
        raise
