url=http://klcl.pku.edu.cn/attach/12973779684875000004-%E7%8E%B0%E4%BB%A3%E6%B1%89%E8%AF%AD%E5%88%87%E5%88%86%E3%80%81%E6%A0%87%E6%B3%A8%E3%80%81%E6%B3%A8%E9%9F%B3%E8%AF%AD%E6%96%99%E5%BA%93-1998%E5%B9%B41%E6%9C%88%E4%BB%BD%E6%A0%B7%E4%BE%8B%E4%B8%8E%E8%A7%84%E8%8C%8320110330.rar
filename=PKU_data
target_name=./datasets/PKU1998_01/data.txt
wget $url -O $filename.rar
unrar x -ad $filename.rar
mkdir -p `dirname $target_name`
mv $filename/04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330/1998-01-2003版-带音.txt $target_name
rm -rf $filename