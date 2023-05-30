[[ $# -ne 2 ]] && echo "Invalid parameters!\nEXPECTED :: ./script input_dir out_dir\nExiting..." && exit -1 

in_dir=$1
out=$2

[[ -d "$out" ]] || (echo "\"$out\" dir not present, creating ..." ; mkdir "$out" )

cnt=0

ls -1 "$in_dir" | while read ln
do
	fc=0
	ls -1 "$in_dir/$ln" | while read f
	do 
		fc=$((fc +1))
		e1=$(echo $f  | rev | cut -d '.' -f 1 | rev )
		
		echo "executing : cp \"$in_dir/$ln/$f\" \"img/img_${fc}_${cnt}.${e1}\""
		cp "$in_dir/$ln/$f" "img/img_${cnt}_${fc}.${e1}"
	done
	cnt=$((cnt+1))
done
