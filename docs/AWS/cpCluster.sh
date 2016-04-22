#Grab all lines with node in the name
ips=$(cat /etc/hosts | awk '/node/ {print $1}')
nodenames="$(cat /etc/hosts | awk '/node/ {print $2}')"

#Copy data
for ip in $ips
do
   #make sure the directory exists
   filename="$(pwd)"/$1
   dirs=$(dirname $filename)
   ssh ec2-user@$ip "mkdir -p \"$dirs\""
   scp $filename ec2-user@$ip:$filename
done


