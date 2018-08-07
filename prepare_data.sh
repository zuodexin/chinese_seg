set -e
datafile=output/merged.txt
python merge_books.py datasets/books
python raw2plain.py
cat  output/data/plain.txt >> $datafile
python raw2std.py