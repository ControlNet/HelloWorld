#!/bin/bash

# Check the validations of command-line argument
if [ $# -ne 0 ]
then
    echo You should not input any command-line argument.
    exit 1
fi

# Ask user to input the dicrectory name
echo Please input a directory name you want.
read name

# Check if the file is exsited or not
if [ ! -d "$name" ]
then
    echo The directory is not existed
    exit 1
fi

# Print the content of chosen directory
echo $name is the directory you choose. The content in it shows below.
ls $name/

# Ask user to input a character to filter files
echo Please input a character you want to search in the end of files.
read char

# Check the character input
if [ ${#char} -ne 1 ]
then
    echo The length of character should be 1
    exit 1
fi

# Print the result
ls -d $name/*$char
exit 0
