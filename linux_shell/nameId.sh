#!/bin/bash

# Check the vadidation of command line argument.
if [ $# -ne 0 ]
then
    echo Error: No command line argument must be used!
    exit 1
fi

# Ask user to input surname.
echo Enter surname:
read surname

# Check the vadidation of surname.
if [ ${#surname} -eq 0 ]
then
    echo Error: Surname must not be blank
    exit 1
elif [ ${#surname} -lt 2 ]
then
    echo Error: Surname should be at least 2 characters
    exit 1
fi

# Ask user to input ID.
echo Enter ID:
read id

# Check the vadidation of ID.
if [ ${#id} -eq 0 ]
then
    echo Error: ID must not be blank
    exit 1
elif [ ${#id} -ne 6 ]
then
    echo Error: ID must be exactly 6 characters
    exit 1
fi

# Print the final result.
echo Thank you, your username is: "${surname:0:2}${id:0:4}"
exit 0
