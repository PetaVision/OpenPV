#Grab all lines with node in the name
ips=$(cat /etc/hosts | awk '/node/ {print $1}')
nodenames="$(cat /etc/hosts | awk '/node/ {print $2}')"

#Copy data
for ip in $ips
do
   scp "$(pwd)"/$1 $ip:"$(pwd)"/$1
done


