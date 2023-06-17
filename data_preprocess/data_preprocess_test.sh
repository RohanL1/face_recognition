#!/bin/bash

help()
{
	echo "Invalid parameters!"
	echo "EXPECTED :: $0 <INPUT_DIR> <OUTPUT_DIR>" 
	echo "Exiting..."
}

out_banner()
{
	echo "AUG_OUT_DIR : $AUG_OUT_DIR"
	echo "COMBINED_OUT_DIR : $COMBINED_OUT_DIR"
	echo "FACE_OUT_DIR : $FACE_OUT_DIR"
	echo "ALIGNED_OUT_DIR : $ALIGNED_OUT_DIR"

}

check_class_folders()
{
CURR_DIR=$1
echo "checking for all the class directorie in $CURR_DIR"
echo "Abdullah Khan
Akshat Kalra
Bowen Cheng
Chakrapani Chitnis
Chin Fang Hsu
Edison Nalluri
Kai Cong
Kyle Fenole
Landis Fusato
Minghao Zhang
Mohit Jawale
Patrick Lee
Peiyuan Li
Rahim Firoz Chunara
Rohan Vikas Lagare
Sadwi Kandula
Samuel Anderson
Shaunak Chaudhary
Tampara Venkata Santosh Anish Dora
Weixuan Lin
Xinyu Dong
Yaoyao Peng
Yufan Lin" | while read SUB_DIR 
do 
CHECK_DIR="${CURR_DIR}/${SUB_DIR}"
[[ -d "$CHECK_DIR" ]] || (echo "\"$CHECK_DIR\" dir not present, creating ..." ; mkdir -p "$CHECK_DIR" )
done
}


#### MAIN 
[[ $# -ne 2 ]] && help && exit -1 

IN_DIR=$1
OUT_DIR=$2

[[ -d "$OUT_DIR" ]] || (echo "\"$OUT_DIR\" dir not present, creating ..." ; mkdir -p "$OUT_DIR" )

AUG_OUT_DIR="${OUT_DIR}/augment/"
COMBINED_OUT_DIR="${OUT_DIR}/combined/"
FACE_OUT_DIR="${OUT_DIR}/face/"
ALIGNED_OUT_DIR="${OUT_DIR}/aligned/"
FINAL_OUT_DIR="${OUT_DIR}/final/"

out_banner

echo "deleting .DS_store files from input dir if present ... "
find ${IN_DIR} -name ".DS_Store" -type f -delete

# echo "ALIGNING IMAGES ..."
# python3 align.py "${IN_DIR}" "${ALIGNED_OUT_DIR}"
# echo "COMPLETED!"

echo "CUTTING FACES FROM IMAGES ..."
python3 cut_face.py  "${IN_DIR}" "${FACE_OUT_DIR}"
echo "COMPLETED!"

check_class_folders "${FACE_OUT_DIR}"

echo "PLEASE CHECK FACES DIR FOR INCORRET FACES AND DELETE"
echo "FACE DIR PATH : '$FACE_OUT_DIR'"
echo "PRESS ENTER TO CONTINUE ..."
read tmp < /dev/tty


echo "COPING ALL IMAGES TO COMBINED FOLDER ..."
./cp_data_by_class.sh  "${FACE_OUT_DIR}" "${COMBINED_OUT_DIR}"
echo "COMPLETED!"

mv "${COMBINED_OUT_DIR}" "${FINAL_OUT_DIR}"

echo "final output dir : ${FINAL_OUT_DIR}"