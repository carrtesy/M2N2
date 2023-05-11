
echo "dir $1 file $2 gpu $3"
for file in ./$1/*/$2; do
	sh $file $3
done
