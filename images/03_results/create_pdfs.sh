for file in $(ls *.svg); do
	bname=$(basename "$file" .svg)
	inkscape -D -z --file=$file --export-pdf=$bname.pdf --export-latex
	done
