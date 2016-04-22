#include "Segmentify.hpp"

namespace PV {

Segmentify::Segmentify(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

Segmentify::Segmentify() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int Segmentify::initialize_base() {
   numChannels = 0;
   originalLayerName = NULL;
   originalLayer = NULL;
   segmentLayerName = NULL;
   segmentLayer = NULL;
   numLabelVals = 0;
   labelIdxBuf = NULL;
   labelVals = NULL;
   labelCount = NULL;

   return PV_SUCCESS;
}

int Segmentify::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int Segmentify::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   ioParam_segmentLayerName(ioFlag);
   ioParam_inputMethod(ioFlag);
   ioParam_outputMethod(ioFlag);
   return status;
}


void Segmentify::ioParam_inputMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inputMethod", &inputMethod);
   if(strcmp(inputMethod, "average") == 0){
   }
   else if(strcmp(inputMethod, "sum") == 0){
   }
   else if(strcmp(inputMethod, "max") == 0){
   }
   else{
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: inputMethod must be \"average\", \"sum\", or \"max\".\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void Segmentify::ioParam_outputMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "outputMethod", &outputMethod);
   if(strcmp(outputMethod, "centroid") == 0){
   }
   else if(strcmp(outputMethod, "fill") == 0){
   }
   else{
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: outputMethod must be \"centriod\" or \"fill\".\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void Segmentify::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
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

void Segmentify::ioParam_segmentLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "segmentLayerName", &segmentLayerName);
   assert(segmentLayerName);
   if (ioFlag==PARAMS_IO_READ && segmentLayerName[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: segmentLayerName must be set.\n",
                 getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

int Segmentify::communicateInitInfo() {
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

   //Get segment layer
   HyPerLayer* tmpLayer = parent->getLayerFromName(segmentLayerName);
   if (tmpLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: segmentLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, segmentLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   segmentLayer = dynamic_cast <SegmentLayer*>(tmpLayer);
   if (segmentLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: segmentLayerName \"%s\" is not a SegmentLayer.\n",
                 getKeyword(), name, segmentLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   if (segmentLayer->getInitInfoCommunicatedFlag()==false) {
      return PV_POSTPONE;
   }

   //Sync with input layer
   originalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(originalLayer);

   //Check sizes
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * segLoc = segmentLayer->getLayerLoc();
   const PVLayerLoc * thisLoc = getLayerLoc();
   assert(srcLoc != NULL && segLoc != NULL);

   //Src layer must have the same number of features as this layer
   if (srcLoc->nf != thisLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayer \"%s\" does not have the same feature dimension as this layer.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nf=%d) versus (nf=%d)\n",
                 srcLoc->nf, thisLoc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   //Segment layer must have 1 feature
   if(segLoc->nf != 1){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: segmentLayer \"%s\" can only have 1 feature.\n",
         getKeyword(), name, segmentLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status;
}

int Segmentify::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   labelToIdx.clear();
   labelVals = (float**) calloc(getLayerLoc()->nf, sizeof(float*));
   labelCount = (int**) calloc(getLayerLoc()->nf, sizeof(int*));
   labelIdxBuf = NULL;
   //Don't allocate inner buffers yet; this will get done based on how many labels are in the current image
   
   return status;
}

int Segmentify::checkLabelValBuf(int newSize){
   if(newSize <= numLabelVals){
      return PV_SUCCESS;
   }

   //Grow buffers
   for(int i = 0; i < getLayerLoc()->nf; i++){
      labelVals[i] = (float*) realloc(labelVals[i], newSize * sizeof(float));
      labelCount[i] = (int*) realloc(labelCount[i], newSize * sizeof(int));
   }
   labelIdxBuf = (int*) realloc(labelIdxBuf, newSize * sizeof(int));

   numLabelVals = newSize;

   return PV_SUCCESS;
}

int Segmentify::allocateV(){
   //Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
   return PV_SUCCESS;
}

int Segmentify::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int Segmentify::initializeActivity() {
   return PV_SUCCESS;
}

int Segmentify::buildLabelToIdx(int batchIdx){
   InterColComm * icComm = parent->icCommunicator();
   int numMpi = icComm->commSize();
   int rank = icComm->commRank();

   labelToIdx.clear();
   //First, we need a single scalar per feature per segment label
   //We need to build a data structure that maps from labels to a vector index
   int numLabels = 0;
   if(rank == 0){
      std::map<int, int> segMap = segmentLayer->getCenterIdxBuf(batchIdx);
      //From the map, we want to grab the set of keys and store it into an int array for broadcasting
      numLabels = segMap.size();
      //Adjust size of buffers
      checkLabelValBuf(numLabels);
      //Fill buffer
      int l = 0;
      for(std::map<int, int>::iterator it = segMap.begin(); it != segMap.end(); ++it){
         labelIdxBuf[l] = it->first; //Store key
         l++;
      }
   }

   //Broadcast number and list of labels from the root process to rest 
   MPI_Bcast(&numLabels, 1, MPI_INT, 0, icComm->communicator());
   checkLabelValBuf(numLabels);
   MPI_Bcast(labelIdxBuf, numLabels, MPI_INT, 0, icComm->communicator());

   for(int l = 0; l < numLabels; l++){
      //Translate the label buffer into the labelToIdx buffer
      labelToIdx[labelIdxBuf[l]] = l;
      //Initialize labelVals based on value reduction type
      //If max, initialize to -inf
      for(int fi = 0; fi < getLayerLoc()->nf; fi++){
         //Set count to 0
         labelCount[fi][l] = 0;
         if(strcmp(inputMethod, "max") == 0){
            labelVals[fi][l] = -INFINITY;
         }
         //If average or sum, initialize to 0
         else if(strcmp(inputMethod, "average") == 0 || strcmp(inputMethod, "sum") == 0){
            labelVals[fi][l] = 0;
         }
         else{
            assert(0); //should never get here
         }
      }
   }
   return PV_SUCCESS;
}

int Segmentify::calculateLabelVals(int batchIdx){
   InterColComm * icComm = parent->icCommunicator();

   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * segLoc = segmentLayer->getLayerLoc();

   assert(segLoc->nf == 1);

   pvdata_t * srcA = originalLayer->getActivity();
   pvdata_t * segA = segmentLayer->getActivity();

   assert(srcA);
   assert(segA);

   pvdata_t * srcBatchA = srcA + batchIdx * originalLayer->getNumExtended();
   pvdata_t * segBatchA = segA + batchIdx * segmentLayer->getNumExtended();

   //Loop through source values
   //As segments are restricted only, we loop through restricted activity
   for(int yi = 0; yi < srcLoc->ny; yi++){ 
      //We caluclate the index into the segment buffer and this buffer based on the
      //relative size differences between source and label buffers
      float segToSrcScaleY = (float)segLoc->ny/(float)srcLoc->ny;
      int segmentYi = round(yi * segToSrcScaleY);
      for(int xi = 0; xi < srcLoc->nx; xi++){
         float segToSrcScaleX = (float)segLoc->nx/(float)srcLoc->nx;
         int segmentXi = round(xi * segToSrcScaleX);
         //Convert segment x and y index into extended linear index into the segment buffer
         int extSegIdx = (segmentYi + segLoc->halo.up) * (segLoc->nx + segLoc->halo.lt + segLoc->halo.rt) + (segmentXi + segLoc->halo.lt);

         //Assuming segments are ints
         int labelVal = round(segBatchA[extSegIdx]);

         //This label should always exist in the map
         //labelIdx is the index into the vals buffer
         int labelIdx = labelToIdx.at(labelVal);

         for(int fi = 0; fi < srcLoc->nf; fi++){
            //Convert restricted yi and xi to extended
            //with resepct to the source
            int extSrcIdx = (yi + srcLoc->halo.up) * (srcLoc->nx + srcLoc->halo.lt + srcLoc->halo.rt) * srcLoc->nf + (xi + srcLoc->halo.lt) * srcLoc->nf + fi;
            float srcVal = srcBatchA[extSrcIdx];
            labelCount[fi][labelIdx]++;
            //Fill labelVals and labelCount
            if(strcmp(inputMethod, "max") == 0){
               if(labelVals[fi][labelIdx] < srcVal){
                  labelVals[fi][labelIdx] = srcVal;
               }
            }
            else if(strcmp(inputMethod, "average") == 0 || strcmp(inputMethod, "sum") == 0){
               labelVals[fi][labelIdx] += srcVal;
            }
         } //End of fi loop
      }//End of xi loop
   }//End of yi loop

   int numLabels = labelToIdx.size();

   int rank = icComm->commRank();

   //We need to reduce our labelVec array
   for(int fi = 0; fi < srcLoc->nf; fi++){
      MPI_Allreduce(MPI_IN_PLACE, labelCount[fi], numLabels, MPI_INT, MPI_SUM, icComm->communicator());
      if(strcmp(inputMethod, "max") == 0){
         MPI_Allreduce(MPI_IN_PLACE, labelVals[fi], numLabels, MPI_FLOAT, MPI_MAX, icComm->communicator());
      }
      else if(strcmp(inputMethod, "sum") == 0 || strcmp(inputMethod, "average") == 0){
         MPI_Allreduce(MPI_IN_PLACE, labelVals[fi], numLabels, MPI_FLOAT, MPI_SUM, icComm->communicator());
      }
      //If average, divide sum by count
      if(strcmp(inputMethod, "average") == 0){
         for(int l = 0; l < numLabels; l++){
            labelVals[fi][l] = labelVals[fi][l] / labelCount[fi][l];
         }
      }
   }

   return PV_SUCCESS;
}

int Segmentify::setOutputVals(int batchIdx){
   //Given the labelVals, we want to fill the output A buffer with what each val should be
   const PVLayerLoc * segLoc = segmentLayer->getLayerLoc();
   const PVLayerLoc * thisLoc = getLayerLoc();

   assert(segLoc->nf == 1);

   pvdata_t * segA = segmentLayer->getActivity();
   pvdata_t * thisA = getActivity();

   assert(thisA);
   assert(segA);

   pvdata_t * segBatchA = segA + batchIdx * segmentLayer->getNumExtended();
   pvdata_t * thisBatchA = thisA + batchIdx * getNumExtended();

   //Reset activity values
   for(int ni = 0; ni < getNumExtended(); ni++){
      thisBatchA[ni] = 0;
   }

   //Scale factors between this layer and segment layer
   float thisToSegScaleX = (float)thisLoc->nx/(float)segLoc->nx;
   float thisToSegScaleY = (float)thisLoc->ny/(float)segLoc->ny;

   //If by centroid, get centroid map from SegmentLayer and set each value
   if(strcmp(outputMethod, "centroid") == 0){
      std::map<int, int> segMap = segmentLayer->getCenterIdxBuf(batchIdx);
      //Centroids are stored in global restricted space, with respect to the segment layer
      for(std::map<int, int>::iterator it = segMap.begin(); it != segMap.end(); ++it){
         int label = it->first;
         int segGlobalResIdx = it->second;
         //Convert to restrictd x and y coords wrt segment layer
         int segGlobalResX = segGlobalResIdx % (segLoc->nxGlobal);
         int segGlobalResY = segGlobalResIdx / (segLoc->nyGlobal);
         //Convert to x and y wrt this layer
         int thisGlobalResX = round(segGlobalResX * thisToSegScaleX);
         int thisGlobalResY = round(segGlobalResY * thisToSegScaleY);
         //If we're within bounds in this process
         if(thisGlobalResX >= thisLoc->kx0 && thisGlobalResX < thisLoc->kx0 + thisLoc->nx &&
            thisGlobalResY >= thisLoc->ky0 && thisGlobalResY < thisLoc->ky0 + thisLoc->ny){
            //Convert thisGlobalResX and Y to an extended local linear index
            int thisLocalExtX = thisGlobalResX - thisLoc->kx0 + thisLoc->halo.lt;
            int thisLocalExtY = thisGlobalResY - thisLoc->ky0 + thisLoc->halo.up;
            for(int fi = 0; fi < thisLoc->nf; fi++){
               int thisLocalExtIdx = thisLocalExtY * (thisLoc->nx + thisLoc->halo.lt + thisLoc->halo.rt) * thisLoc->nf + thisLocalExtX * thisLoc->nf + fi;
               //Set value based on labelVals
               thisBatchA[thisLocalExtIdx] = labelVals[fi][labelToIdx.at(label)];
            }
         }
      }
   }
   else if(strcmp(outputMethod, "fill") == 0){
      //Loop through this layer's neurons
      //Looping through restricted
      for(int yi = 0; yi < thisLoc->ny; yi++){
         //Translate from this yi to segment's yi 
         int segResY = round((float)yi / (float)thisToSegScaleY);
         for(int xi = 0; xi < thisLoc->nx; xi++){
            int segResX = round((float)xi / (float)thisToSegScaleX);
            //Convert restricted segment index to extended
            int segExtIdx = (segResY + segLoc->halo.up) * (segLoc->nx + segLoc->halo.lt + segLoc->halo.rt) + (segResX + segLoc->halo.lt);
            //Get label based on segment layer
            int label = round(segBatchA[segExtIdx]);
            //Fill index with value from labelVals;
            for(int fi = 0; fi < thisLoc->nf; fi++){
               //Calulate ext index
               int thisExtIdx = (yi + thisLoc->halo.up) * (thisLoc->nx + thisLoc->halo.lt + thisLoc->halo.rt) * thisLoc->nf + (xi + thisLoc->halo.lt) * thisLoc->nf + fi;
               thisBatchA[thisExtIdx] = labelVals[fi][labelToIdx.at(label)];
            }
         }
      }
   }
   return PV_SUCCESS;
}


int Segmentify::updateState(double timef, double dt) {
   int status;

   //Using the segment activity, we want to compress all values within a segment to a single value (per feature)
   for (int bi = 0; bi < getLayerLoc()->nbatch; bi++){
      buildLabelToIdx(bi);
      calculateLabelVals(bi);
      setOutputVals(bi);
   }
   
   return status;
}


Segmentify::~Segmentify() {
   free(originalLayerName);
   clayer->V = NULL;
}

BaseObject * createSegmentify(char const * name, HyPerCol * hc) {
   return hc ? new Segmentify(name, hc) : NULL;
}

} /* namespace PV */
