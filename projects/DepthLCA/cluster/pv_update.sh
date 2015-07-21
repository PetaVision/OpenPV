#Grab all lines with node in the name
nodenames=$(cat ./nodefile | awk '/ / {print $1}')

#For each ip address
for node in $nodenames
do
   if [ "$node" == "persona" ]
   then
      #Update and build petavision
      ssh -t $node 'source ~/.bash_profile; cd ~/workspace/PetaVision; svn update; cd ~/workspace/; /usr/local/bin/cmake -DCLANG_OMP=True -DCMAKE_BUILD_TYPE=Release -DCUDA_GPU=True -DCUDA_RELEASE=True -DCUDNN=True -DCUDNN_PATH=~/cudnn -DOPEN_MP_THREADS=True -DPV_DIR=~/workspace/PetaVision; cd ~/workspace/PetaVision; make -j 8'
   else
      ssh -t $node 'source ~/.bash_profile; cd ~/workspace/PetaVision; svn update; cd ~/workspace/; cp ~/workspace/PetaVision/docs/cmake/CMakeLists.txt .; /usr/local/bin/cmake -DCLANG_OMP=True -DCMAKE_BUILD_TYPE=Release -DCUDA_GPU=True -DCUDA_RELEASE=True -DCUDNN=True -DCUDNN_PATH=~/cudnn -DOPEN_MP_THREADS=True -DPV_DIR=~/workspace/PetaVision; cd ~/workspace/PetaVision; make -j 8'
   fi

done
