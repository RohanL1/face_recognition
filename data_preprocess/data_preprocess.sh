#!/bin/bash

help()
{
	echo "Invalid parameters!"
	echo "EXPECTED :: $0 <INPUT_DIR> <OUTPUT_DIR>" 
	echo "Exiting..."
}

[[ $# -ne 2 ]] && help && exit -1 

in_dir=$1
out_dir=$2

[[ -d "$out_dir" ]] || (echo "\"$out_dir\" dir not present, creating ..." ; mkdir "$out_dir" )

align.py
cut_face.py
data_aug_new.py