#Generate new key
ssh-keygen -t rsa -f clusterkey -q -N ""

#Grab all lines with node in the name
ips=$(cat ./hosts | awk '/node/ {print $1}')
nodenames="$(cat ./hosts | awk '/node/ {print $2}')"

counter=0
#For each ip address
for ip in $ips
do
   counter=$((counter+1))
   #name=$(echo "$nodenames" | sed -n ${counter}p)

   #Copy hosts file to home directory
   scp ./hosts ec2-user@$ip:~/
   ssh -t ec2-user@$ip 'sudo mv ~/hosts /etc/hosts'
   #Copy private key to ~/.ssh/id_rsa
   scp ./clusterkey ec2-user@$ip:~/
   ssh -t ec2-user@$ip 'sudo mv ~/clusterkey ~/.ssh/id_rsa'
   #Append public key to authorized_keys
   scp ./clusterkey.pub ec2-user@$ip:~/
   ssh -t ec2-user@$ip 'cat ~/clusterkey.pub >> ~/.ssh/authorized_keys'

   for name in $nodenames
   do
      ssh ec2-user@$ip 'ssh-keyscan '$name' >> ~/.ssh/known_hosts' 
   done

   #Copy node file to home directory
   scp ./nodefile ec2-user@$ip:~/
done
