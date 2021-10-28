#!/bin/bash
FOLDERNAME=/usr/bin
FILETYPE="shell script"
file "$FOLDERNAME"/* | grep "$FILETYPE" | wc -l
