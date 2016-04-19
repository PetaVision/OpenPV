#include "SegmentLayer.hpp"

namespace PV {

SegmentLayer::SegmentLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

SegmentLayer::SegmentLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int SegmentLayer::initialize_base() {
   segmentMethod = NULL;
   originalLayerName = NULL;
   numChannels = 0;
   labelBufSize = 0;
   labelBuf = NULL;
   maxXBuf = NULL;
   maxYBuf = NULL;
   minXBuf = NULL;
   minYBuf = NULL;
   centerIdxBufSize = 0;
   centerIdxBuf = NULL;
   allLabelsBuf = NULL;
   
   return PV_SUCCESS;
}

int SegmentLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int SegmentLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_segmentMethod(ioFlag);
   ioParam_originalLayerName(ioFlag);
   return status;
}

void SegmentLayer::ioParam_segmentMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "segmentMethod", &segmentMethod);
   assert(segmentMethod);
   //Check valid segment methods
   //none means the gsyn is already a segmentation. Helpful if reading segmentation from pvp
   if(strcmp(segmentMethod, "none") == 0){
   }
   //TODO add in other segmentation methods
   //How do we segment across MPI margins?
   else{
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: segmentMethod %s not recognized. Current options are \"none\".\n",
                 getKeyword(), name, segmentMethod);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void SegmentLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
   assert(originalLayerName);
   if (ioFlag==PARAMS_IO_READ && originalLayerName[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

int SegmentLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   //Get original layer
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (originalLayer->getInitInfoCommunicatedFlag()==false) {
      return PV_POSTPONE;
   }

   //Sync margins
   originalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(originalLayer);

   //Check size
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * thisLoc = getLayerLoc();

   //Original layer must be the same x/y size as this layer
   if (srcLoc->nxGlobal != thisLoc->nxGlobal || srcLoc->nyGlobal != thisLoc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayer \"%s\" does not have the same x and y dimensions as this layer.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, thisLoc->nxGlobal, thisLoc->nyGlobal);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   //This layer must have only 1 feature
   if(thisLoc->nf != 1){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: SegmentLayer must have 1 feature.\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   //If segmentMethod is none, we also need to make sure the srcLayer also has nf == 1
   if(strcmp(segmentMethod, "none") == 0 && srcLoc->nf != 1){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: Source layer must have 1 feature with segmentation method \"none\".\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status;
}

int SegmentLayer::checkLabelBufSize(int newSize){
   if(newSize <= labelBufSize){
      return PV_SUCCESS;
   }

   const PVLayerLoc* loc = getLayerLoc();

   //Grow buffer
   labelBuf = (int*) realloc(labelBuf, newSize*sizeof(int));
   maxXBuf = (int*) realloc(maxXBuf, newSize*sizeof(int));
   maxYBuf = (int*) realloc(maxYBuf, newSize*sizeof(int));
   minXBuf = (int*) realloc(minXBuf, newSize*sizeof(int));
   minYBuf = (int*) realloc(minYBuf, newSize*sizeof(int));

   //Set new size
   labelBufSize = newSize;
   return PV_SUCCESS;
}

int SegmentLayer::loadLabelBuf(){
   //Load in maxX and label buf from maxX map
   int numLabels = maxX.size();
   //Allocate send buffer to the right size
   checkLabelBufSize(numLabels);

   int idx = 0;
   for(std::map<int, int>::iterator it = maxX.begin();
         it != maxX.end(); ++it){
      labelBuf[idx] = it->first; //Store key in label
      maxXBuf[idx] = it->second; //Store vale in maxXBuf
      idx++;
   }
   assert(idx == numLabels);

   //Load rest of buffers based on label
   for(int i = 0; i < numLabels; i++){
      int label = labelBuf[i];
      maxYBuf[i] = maxY.at(label);
      minXBuf[i] = minX.at(label);
      minYBuf[i] = minY.at(label);
   }
   return PV_SUCCESS;
}

int SegmentLayer::loadCenterIdxMap(int batchIdx, int numLabels){
   for(int i = 0; i < numLabels; i++){
      int label = allLabelsBuf[i];
      int idx = centerIdxBuf[i];
      centerIdx[batchIdx][label] = idx;
   }
   return PV_SUCCESS;
}

int SegmentLayer::checkIdxBufSize(int newSize){
   if(newSize <= centerIdxBufSize){
      return PV_SUCCESS;
   }

   //Grow buffer
   centerIdxBuf = (int*) realloc(centerIdxBuf, newSize*sizeof(int));
   allLabelsBuf = (int*) realloc(allLabelsBuf , newSize*sizeof(int));
   //Set new size
   centerIdxBufSize = newSize;
   return PV_SUCCESS;

}

int SegmentLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   int nbatch = getLayerLoc()->nbatch;
   //growLabelBuffers(16); //Initialize with labels
   maxX.clear();
   maxY.clear();
   minX.clear();
   minY.clear();
   centerIdx.clear();

   //Initialize vector of maps
   for(int b = 0; b < nbatch; b++){
      centerIdx.push_back(std::map<int, int>());
   }

   return status;
}

int SegmentLayer::allocateV(){
   //Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
   return PV_SUCCESS;
}

int SegmentLayer::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int SegmentLayer::initializeActivity() {
   return PV_SUCCESS;
}

int SegmentLayer::updateState(double timef, double dt) {
   pvdata_t* srcA = originalLayer->getActivity();
   pvdata_t* thisA = getActivity();
   assert(srcA);
   assert(thisA);

   const PVLayerLoc* loc = getLayerLoc();

   //Segment input layer based on segmentMethod
   if(strcmp(segmentMethod, "none") == 0){
      int numBatchExtended = getNumExtendedAllBatches();
      //Copy activity over
      //Since both buffers should be identical size, we can do a memcpy here
      memcpy(thisA, srcA, numBatchExtended * sizeof(pvdata_t));
   }
   else{
      //This case should never happen
      assert(0);
   }

   assert(loc->nf == 1);

   //Clear centerIdxs
   for(int bi = 0; bi < loc->nbatch; bi++){
      centerIdx[bi].clear();
   }

   for(int bi = 0; bi < loc->nbatch; bi++){
      pvdata_t* batchA = thisA + bi * getNumExtended();
      //Reset max/min buffers
      maxX.clear();
      maxY.clear();
      minX.clear();
      minY.clear();

      //Loop through this buffer to fill labelVec and idxVec
      //Looping through restricted, but indices are extended
      for(int yi = loc->halo.up; yi < loc->ny+loc->halo.up; yi++){
         for(int xi = loc->halo.lt; xi < loc->nx+loc->halo.lt; xi++){
            //Convert to local extended linear index
            int niLocalExt = yi * (loc->nx+loc->halo.lt+loc->halo.rt) + xi;
            //Convert yi and xi to global res index
            int globalResYi = yi - loc->halo.up + loc->ky0;
            int globalResXi = xi - loc->halo.lt + loc->kx0;

            //Get label value
            //Note that we're assuming that the activity here are integers,
            //even though the buffer is floats
            int labelVal = round(batchA[niLocalExt]);

            //Calculate max/min x and y for a single batch
            //If labelVal exists in map
            if(maxX.count(labelVal)){
               //Here, we're assuming the 4 maps are in sync, so we use the 
               //.at method, as it will throw an exception as opposed to the 
               //[] operator, which will simply add the key into the map
               if(globalResXi > maxX.at(labelVal)){
                  maxX[labelVal] = globalResXi;
               }
               if(globalResXi < minX.at(labelVal)){
                  minX[labelVal] = globalResXi;
               }
               if(globalResYi > maxY.at(labelVal)){
                  maxY[labelVal] = globalResYi;
               }
               if(globalResYi < minY.at(labelVal)){
                  minY[labelVal] = globalResYi;
               }
            }
            //If doesn't exist, add into map with current vals
            else{
               maxX[labelVal] = globalResXi;
               minX[labelVal] = globalResXi;
               maxY[labelVal] = globalResYi;
               minY[labelVal] = globalResYi;
            }
         }
      }

      //We need to mpi across processors in case a segment crosses an mpi boundary
      InterColComm * icComm = parent->icCommunicator();
      int numMpi = icComm->commSize();
      int rank = icComm->commRank();

      //Local comm rank
      //Non root processes simply send buffer size and then buffers
      int numLabels = maxX.size();

      if(rank != 0){
         //Load buffers
         loadLabelBuf();
         //Send number of labels first
         MPI_Send(&numLabels, 1, MPI_INT, 0, rank, icComm->communicator());
         //Send labels, then max/min buffers
         MPI_Send(labelBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(maxXBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(maxYBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(minXBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(minYBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());

         //Receive the full centerIdxBuf from root process
         int numCenterIdx = 0;
         MPI_Bcast(&numCenterIdx, 1, MPI_INT, 0, icComm->communicator());
         checkIdxBufSize(numCenterIdx);

         MPI_Bcast(allLabelsBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(centerIdxBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());

         //Load buffer into centerIdx map
         loadCenterIdxMap(bi, numCenterIdx);
      }
      //Root process stores everything
      else{
         //One recv per buffer
         for(int recvRank = 1; recvRank < numMpi; recvRank++){
            int numRecvLabels = 0;
            MPI_Recv(&numRecvLabels, 1, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            checkLabelBufSize(numRecvLabels);

            MPI_Recv(labelBuf, numRecvLabels, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            MPI_Recv(maxXBuf, numRecvLabels, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            MPI_Recv(maxYBuf, numRecvLabels, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            MPI_Recv(minXBuf, numRecvLabels, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            MPI_Recv(minYBuf, numRecvLabels, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);

            for(int i = 0; i < numRecvLabels; i++){
               int label = labelBuf[i];
               //Add on to maps
               //If the label already exists, fill with proper max/min
               if(maxX.count(label)){
                  if(maxXBuf[i] > maxX.at(label)){
                     maxX[label] = maxXBuf[i];
                  }
                  if(maxYBuf[i] > maxY.at(label)){
                     maxY[label] = maxYBuf[i];
                  }
                  if(minXBuf[i] < minX.at(label)){
                     minX[label] = minXBuf[i];
                  }
                  if(minYBuf[i] < minY.at(label)){
                     minY[label] = minYBuf[i];
                  }
               }
               else{
                  maxX[label] = maxXBuf[i];
                  maxY[label] = maxYBuf[i];
                  minX[label] = minXBuf[i];
                  minY[label] = minYBuf[i];
               }
            }
         }

         //Maps are now filled with all segments from the image
         //Fill centerIdx based on max/min
         for(std::map<int, int>::iterator it = maxX.begin();
               it != maxX.end(); ++it){
            int label = it->first;
            int centerX = minX.at(label) + (maxX.at(label) - minX.at(label))/2;
            int centerY = minY.at(label) + (maxY.at(label) - minY.at(label))/2;
            //Convert centerpoints (in global res idx) to linear idx (in global res space)
            int centerIdxVal = centerY * (loc->nxGlobal) + centerX;
            //Add to centerIdxMap
            centerIdx[bi][label] = centerIdxVal;
         }

         //Fill centerpoint buffer
         int numCenterIdx = centerIdx[bi].size();
         checkIdxBufSize(numCenterIdx);

         int idx = 0;
         for(std::map<int, int>::iterator it = centerIdx[bi].begin(); 
               it != centerIdx[bi].end(); ++it){
            allLabelsBuf[idx] = it->first;
            centerIdxBuf[idx] = it->second;
            idx++;
         }

         //Broadcast buffers
         MPI_Bcast(&numCenterIdx, 1, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(allLabelsBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(centerIdxBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
      }
   } //End batch loop
   
   //centerIdx now stores each center coordinate of each segment
   return PV_SUCCESS;
}

SegmentLayer::~SegmentLayer() {
   free(originalLayerName);
   clayer->V = NULL;
   maxX.clear();
   maxY.clear();
   minX.clear();
   minY.clear();
   //This should call destructors of all maps within the vector
   centerIdx.clear();
}

BaseObject * createSegmentLayer(char const * name, HyPerCol * hc) {
   return hc ? new SegmentLayer(name, hc) : NULL;
}

} /* namespace PV */
