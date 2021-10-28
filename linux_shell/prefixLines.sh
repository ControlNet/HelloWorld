#!/bin/bash

echo Please enter the file name you want.
read file
count=0
for line in `cat $file`
do
    count=`expr $count + 1`
    echo "$count". "$line"
done
exit 0 
