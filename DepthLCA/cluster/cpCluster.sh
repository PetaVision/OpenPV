#Grab all lines with node in the name
nodenames=$(cat ~/workspace/DepthLCA/cluster/nodefile | awk '/ / {print $1}')

#Copy data
for node in $nodenames
do
   if [ "$node" != "persona" ]
   then
      #make sure the directory exists
      filename="$(pwd)"/$1
      dirs=$(dirname $filename)

      ssh $node "mkdir -p \"$dirs\""
      scp $filename $node:$filename
   fi
done


