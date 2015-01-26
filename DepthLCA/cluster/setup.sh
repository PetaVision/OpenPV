#Generate new key
ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""

#Grab all lines with node in the name
nodenames=$(cat ./nodefile | awk '/ / {print $1}')

#For each ip address
for node in $nodenames
do
   if test -z "$(ssh-keygen -F $node)"; then
      $(ssh-keyscan $node >> ~/.ssh/known_hosts)
   fi

   #Add keys
   ssh -t $node 'mkdir -p ~/.ssh'
   if [ "$node" != "persona" ]
   then
      scp ~/.ssh/id_rsa* $node:~/.ssh/
   fi
   ssh -t $node 'cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys'

   #Put all keys to known_hosts
   for name in $nodenames
   do
      ssh $node 'ssh-keyscan '$name' >> ~/.ssh/known_hosts' 
   done
done
