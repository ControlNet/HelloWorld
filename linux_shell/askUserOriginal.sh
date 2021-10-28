#!/bin/bash

echo Please input a directory name you want.
read name
echo $name is the directory you choose. The content in it shows below.
ls $name/
echo Please input a character you want to search in the end of files.
read char
ls -d $name/*$char
