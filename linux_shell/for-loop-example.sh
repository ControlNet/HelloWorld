#!/bin/bash

# Ask user to input the file name for input
echo Please input the input file name
read inputFile

# Check if the input file is existed
if [ ! -f "$inputFile" ]
then
    echo The input file is not existed!
    exit 1
fi

# Ask user to input the file name for output
echo Please input the output file name
read outputFile

# Check if the output file is existed and ask user to confirm rewrite or not.
while true
do
    if [ -f "$outputFile" ]
    then
        echo The file is existed, do you want to rewrite? '[y/n]'
        read option
    fi

    case "$option" in
      n|N)
        exit 0
	  ;;
      y|Y)
        # Calculate the sum.
        sum=0
        for value in `cat "$inputFile"`
        do
            sum=`expr "$value" + "$sum"`
        done
 
        # Print the result and output the result to the output file.
        echo The sum is "$sum"
        echo "$sum" > "$outputFile"
        echo The result has been export to "$outputFile"
        exit 0
	  ;;
      *)
	    echo Please input correct options.
	  ;;
    esac
done

