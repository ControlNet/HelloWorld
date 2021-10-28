#!/bin/bash

# Check the input validations.
if [ $# -ne 2 ]
then
    echo The number of arguments should be 2.
    exit 1
elif [ ! -f $1 ]
then
    echo The file is not existed.
    exit 1
fi

# Compared the numbers.
line=`cat $1|wc -l`
if [ $2 -eq $line ]
then
    echo You guessed the correct file length!
elif [ $2 -lt $line ]
then
    echo File is longer
else
    echo File is shorter
fi
exit 0
