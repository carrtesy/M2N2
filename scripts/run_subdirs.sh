
echo "$1 @gpu$2"
for file in ./$1/*.sh; do
	sh $file $2
done
