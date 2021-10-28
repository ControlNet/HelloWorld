#!/bin/bash

while true
do
    echo "Please choose an option below."
    echo "1. Add 2 numbers (integers)"
    echo "2. Multiply 2 numbers (integers)"
    echo "3. Display a file"
    echo "4. Exit"
    read option
    case "$option" in
      1)
	# add 2 numbers 
        echo Please enter the first integer you want
        read arg1
	    echo Please enter the second integer you want
        read arg2
        echo The sum of these two integers is `expr "$arg1" + "$arg2"`
        ;;
      2)
	# multiply 2 numbers
        echo Please enter the first integer you want
        read arg1
	    echo Please enter the second integer you want
        read arg2
        echo The product of these two integers is `expr "$arg1" \* "$arg2"`
        ;;
      3)
	# display a file
        echo Please enter the file name you want
        read fileName
        less "$fileName"
        ;;
      4)
	# exit
	echo Thank you for your using this program
	exit 0
        ;;
      *)
	echo Please input a correct option.
	;;
    esac
done

    
