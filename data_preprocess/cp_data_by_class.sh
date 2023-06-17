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

cnt=0

ls -1 "$in_dir" | while read ln
do
	fc=0
	ls -1 "$in_dir/$ln" | while read f
	do 
		fc=$((fc +1))
		e1=$(echo $f  | rev | cut -d '.' -f 1 | rev )
		
		echo "executing : cp \"$in_dir/$ln/$f\" \"${out_dir}/img_${fc}_${cnt}.${e1}\""
		cp "$in_dir/$ln/$f" "${out_dir}/img_${cnt}_${fc}.${e1}"
	done
	cnt=$((cnt+1))
done
