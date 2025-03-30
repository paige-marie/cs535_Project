#!/bin/bash

# I recommend executing this script in a tmux as it takes a while
# you can use monitor-download.sh to monitor the download from outside the tmux
# download is complete when you have 1054 files

TIFS_LIST='Iowa-download-list.txt'

# to get a token, first make an account at https://appeears.earthdatacloud.nasa.gov/
# then execute: `curl --request POST --user <username>:<password> --header "Content-Length: 0" "https://appeears.earthdatacloud.nasa.gov/api/login" > token`
# your token will be in the newly generated token file, pass that token as an argument to this script
# usage: ./download-tifs.sh <token>

# the tifs are placed in a `tifs` subdirectory

if [[ -n $1 ]]; then
	TOKEN=$1
	mkdir -p tifs
	wget --header "Authorization: Bearer $TOKEN" -i $TIFS_LIST -P ./tifs/
else
	echo "Must pass your auth token as an argument"
fi

