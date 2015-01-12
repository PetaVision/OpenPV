if test -n "$1"
then
   all_files=$(ls */CMakeLists.txt)
   for file in $all_files
   do
      echo cp $1 $file
      cp $1 $file
   done
else
   echo "Usage: copyAllCMakeLIsts.sh inputFile"
fi


