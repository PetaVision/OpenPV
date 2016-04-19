/*
 * privateTransposeConn.cpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#include "privateTransposeConn.hpp"

namespace PV {

privateTransposeConn::privateTransposeConn(const char * name, HyPerCol * hc, HyPerConn * parentConn, bool needWeights) {
   int status = initialize(name, hc, parentConn, needWeights);
}

privateTransposeConn::~privateTransposeConn() {
}  // privateTransposeConn::~privateTransposeConn()

//privateTransposeConn initialize will be called during parentConn's communicate
int privateTransposeConn::initialize(const char * name, HyPerCol * hc, HyPerConn * parentConn, bool needWeights) {
   int status = BaseObject::initialize(name, hc); // Don't call HyPerConn::initialize
   assert(status == PV_SUCCESS);

   this->weightInitializer = NULL;
   this->normalizer = NULL;

   this->io_timer     = new Timer(getName(), "postConn", "io     ");
   this->update_timer = new Timer(getName(), "postConn", "update ");

   postConn = parentConn;

   //Pre/post is swapped from orig conn
   pre = postConn->getPost();
   post = postConn->getPre();

   sharedWeights = postConn->usingSharedWeights();
   numAxonalArborLists = postConn->numberOfAxonalArborLists();
   plasticityFlag = false;
   weightUpdatePeriod = 1;
   weightUpdateTime = parentConn->weightUpdateTime;
   shrinkPatches_flag = false;
   //Cannot shrink patches with privateTransposeConn
   assert(parentConn->getShrinkPatches_flag() == false);

   triggerFlag = parentConn->triggerFlag;
   triggerLayer = parentConn->triggerLayer;
   triggerOffset = parentConn->triggerOffset;

   status = setPatchSize();
   assert(status == PV_SUCCESS);
   status = checkPatchDimensions();
   assert(status == PV_SUCCESS);

   this->needAllocWeights = needWeights;

   //Set parentConn's pre and post connections
   return status;
}

//Private transpose conn will have no parameters, will be set by parent connection
int privateTransposeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   std::cout << "Fatal Error, privateTransposeConn ioParamsFillGroup called\n";
   exit(-1);
   return PV_SUCCESS;
}


int privateTransposeConn::communicateInitInfo() {
   //Should never be called
   std::cout << "Fatal Error, privateTransposeConn communicate called\n";
   exit(-1);
   return PV_SUCCESS;
}

int privateTransposeConn::setPatchSize() {
   // If postConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if postConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(postConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int nxp_orig = postConn->xPatchSize();
   int nyp_orig = postConn->yPatchSize();
   nxp = nxp_orig;
   if(xscaleDiff > 0 ) {
      nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
      nxp /= (int) pow(2,-xscaleDiff);
      assert(nxp_orig==nxp*pow( 2, (float) (-xscaleDiff) ));
   }

   int yscaleDiff = pre->getYScale() - post->getYScale();
   nyp = nyp_orig;
   if(yscaleDiff > 0 ) {
      nyp *= (int) pow( 2, yscaleDiff );
   }
   else if(yscaleDiff < 0) {
      nyp /= (int) pow(2,-yscaleDiff);
      assert(nyp_orig==nyp*pow( 2, (float) (-yscaleDiff) ));
   }

   nfp = post->getLayerLoc()->nf;
   // post->getLayerLoc()->nf must be the same as postConn->preSynapticLayer()->getLayerLoc()->nf.
   // This requirement is checked in communicateInitInfo

   //parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   //parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   //parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;

}

int privateTransposeConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   assert(status==PV_SUCCESS);
   normalizer = NULL;
   
   // normalize_flag = false; // replaced by testing whether normalizer!=NULL
   return PV_SUCCESS;
}

int privateTransposeConn::constructWeights(){
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();
   int status = PV_SUCCESS;

   //assert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   
   // createArbors() uses the value of shrinkPatches.  It should have already been read in ioParamsFillGroup.
   //allocate the arbor arrays:
   createArbors();

   setPatchStrides();

   ////allocate weight patches and axonal arbors for each arbor
   ////Allocate all the weights
   //bool is_pooling_from_pre_perspective = (((getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING) || (getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING)) && (!updateGSynFromPostPerspective));
   if (needAllocWeights){
     wDataStart[0] = allocWeights(nPatches, nxp, nyp, nfp);
     assert(this->get_wDataStart(0) != NULL);
   }
   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      status = createWeights(wPatches, arborId);
      assert(wPatches[arborId] != NULL);

      if (needAllocWeights){
         if (arborId > 0){  // wDataStart already allocated
            wDataStart[arborId] = (this->get_wDataStart(0) + sp * nPatches * arborId);
            assert(this->wDataStart[arborId] != NULL);
         }
      }
      if (shrinkPatches_flag || arborId == 0){
         status |= adjustAxonalArbors(arborId);
      }
   }  // arborId

   //call to initializeWeights moved to BaseConnection::initializeState()
   status |= initPlasticityPatches();
   assert(status == 0);
   if (shrinkPatches_flag) {
      for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
         shrinkPatches(arborId);
      }
   }

   return status;
}

//Called in allocateDataStructures, not needed for privateTransposeConn
int privateTransposeConn::initializeDelays(const float * fDelayArray, int size){
   return PV_SUCCESS;
}

int privateTransposeConn::setInitialValues() {
   int status = HyPerConn::setInitialValues(); // calls initializeWeights
   return status;
}

PVPatch*** privateTransposeConn::initializeWeights(PVPatch*** patches, pvwdata_t** dataStart) {
   // privateTransposeConn must wait until after postConn has been normalized, so weight initialization doesn't take place until HyPerCol::run calls finalizeUpdate
   return patches;
}

bool privateTransposeConn::needUpdate(double timed, double dt) {
   return plasticityFlag && postConn->getLastUpdateTime() > lastUpdateTime;
}

int privateTransposeConn::updateState(double time, double dt) {
   assert(plasticityFlag && postConn->getLastUpdateTime() > lastUpdateTime); // should only be called if needUpdate returned true this timestep
   // privateTransposeConn must wait until finalizeUpdate, after normalizers are called,
   // so that it will see the correct weights when it calls transpose.
   return PV_SUCCESS;
}

double privateTransposeConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // privateTransposeConn does not use weightUpdateTime to determine when to update
}

int privateTransposeConn::finalizeUpdate(double time, double dt) {
   int status = PV_SUCCESS;
   update_timer->start();
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
      status = transpose(arborId);  // Apply changes in weights
      if (status==PV_BREAK) { break; }
      assert(status==PV_SUCCESS);
   }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if(postConn->getAllocPostDeviceWeights()){
      updateDeviceWeights();
   }
#endif

   update_timer->stop();
   return status;
}

int privateTransposeConn::transpose(int arborId) {
   if(!needAllocWeights) return PV_SUCCESS;
   return sharedWeights ? transposeSharedWeights(arborId) : transposeNonsharedWeights(arborId);
}

int privateTransposeConn::transposeNonsharedWeights(int arborId) {
   assert(usingSharedWeights()==false);
   const PVLayerLoc * preLocOrig = postConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLocOrig = postConn->postSynapticLayer()->getLayerLoc();
   const PVLayerLoc * preLocTranspose = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLocTranspose = postSynapticLayer()->getLayerLoc();
#ifdef PV_USE_MPI
   InterColComm * icComm = parent->icCommunicator();
   pvwdata_t * sendbuf[NUM_NEIGHBORHOOD];
   pvwdata_t * recvbuf[NUM_NEIGHBORHOOD];
   int size[NUM_NEIGHBORHOOD];
   int startx[NUM_NEIGHBORHOOD];
   int starty[NUM_NEIGHBORHOOD];
   int stopx[NUM_NEIGHBORHOOD];
   int stopy[NUM_NEIGHBORHOOD];
   int blocksize[NUM_NEIGHBORHOOD];
   MPI_Request request[NUM_NEIGHBORHOOD];
   size_t buffersize[NUM_NEIGHBORHOOD];
   bool hasRestrictedNeighbor[NUM_NEIGHBORHOOD];
   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      hasRestrictedNeighbor[neighbor] = neighbor!=LOCAL &&
                                        icComm->neighborIndex(parent->columnId(), neighbor)>=0 &&
                                        icComm->reverseDirection(parent->columnId(), neighbor) + neighbor == NUM_NEIGHBORHOOD;
   }
   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      if (hasRestrictedNeighbor[neighbor]==false ) {
         size[neighbor] = 0;
         startx[neighbor] = -1;
         starty[neighbor] = -1;
         stopx[neighbor] = -1;
         stopy[neighbor] = -1;
         blocksize[neighbor] = 0;
         buffersize[neighbor] = (size_t) 0;
         sendbuf[neighbor] = NULL;
         recvbuf[neighbor] = NULL;
         request[neighbor] = NULL;
      }
      else {
         mpiexchangesize(neighbor,  &size[neighbor], &startx[neighbor], &stopx[neighbor], &starty[neighbor], &stopy[neighbor], &blocksize[neighbor], &buffersize[neighbor]);
         sendbuf[neighbor] = (pvwdata_t *) malloc(buffersize[neighbor]);
         if (sendbuf[neighbor]==NULL) {
            fprintf(stderr, "%s \"%s\": Rank %d process unable to allocate memory for Transpose send buffer: %s\n", this->getKeyword(), name, parent->columnId(), strerror(errno));
            exit(EXIT_FAILURE);
         }
         recvbuf[neighbor] = (pvwdata_t *) malloc(buffersize[neighbor]);
         if (recvbuf[neighbor]==NULL) {
            fprintf(stderr, "%s \"%s\": Rank %d process unable to allocate memory for Transpose receive buffer: %s\n", this->getKeyword(), name, parent->columnId(), strerror(errno));
            exit(EXIT_FAILURE);
         }
         request[neighbor] = NULL;
      }
   }

   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      if (!hasRestrictedNeighbor[neighbor]) { continue; }
      int nbrIdx = icComm->neighborIndex(parent->columnId(), neighbor);
      assert(nbrIdx>=0); // If neighborIndex is negative, there is no neighbor in that direction so hasRestrictedNeighbor should be false

      char * b = (char *) sendbuf[neighbor];
      for (int y=starty[neighbor]; y<stopy[neighbor]; y++) {
         for (int x=startx[neighbor]; x<stopx[neighbor]; x++) {
            int idxExt = kIndex(x,y,0,preLocOrig->nx+preLocOrig->halo.lt+preLocOrig->halo.rt,preLocOrig->ny+preLocOrig->halo.dn+preLocOrig->halo.up,preLocOrig->nf);
            int xGlobalExt = x+preLocOrig->kx0;
            int xGlobalRes = xGlobalExt-preLocOrig->halo.lt;
            int yGlobalExt = y+preLocOrig->ky0;
            int yGlobalRes = yGlobalExt-preLocOrig->halo.up;
            if (xGlobalRes >= preLocOrig->kx0 && xGlobalRes < preLocOrig->kx0+preLocOrig->nx && yGlobalRes >= preLocOrig->ky0 && yGlobalRes < preLocOrig->ky0+preLocOrig->ny) {
               fprintf(stderr, "Rank %d, connection \"%s\", x=%d, y=%d, neighbor=%d: xGlobalRes = %d, preLocOrig->kx0 = %d, preLocOrig->nx = %d\n", parent->columnId(), name, x, y, neighbor, xGlobalRes, preLocOrig->kx0, preLocOrig->nx);
               fprintf(stderr, "Rank %d, connection \"%s\", x=%d, y=%d, neighbor=%d: yGlobalRes = %d, preLocOrig->ky0 = %d, preLocOrig->ny = %d\n", parent->columnId(), name, x, y, neighbor, yGlobalRes, preLocOrig->ky0, preLocOrig->ny);
               exit(EXIT_FAILURE);
            }
            int idxGlobalRes = kIndex(xGlobalRes, yGlobalRes, 0, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
            memcpy(b, &idxGlobalRes, sizeof(idxGlobalRes));
            b += sizeof(idxGlobalRes);
            PVPatch * patchOrig = postConn->getWeights(idxExt, arborId);
            memcpy(b, patchOrig, sizeof(*patchOrig));
            b += sizeof(*patchOrig);
            int postIdxRes = (int) postConn->getGSynPatchStart(idxExt, arborId);
            int postIdxExt = kIndexExtended(postIdxRes, postLocOrig->nx, postLocOrig->ny, postLocOrig->nf, postLocOrig->halo.lt, postLocOrig->halo.rt, postLocOrig->halo.dn, postLocOrig->halo.up);
            int postIdxGlobalRes = globalIndexFromLocal(postIdxRes, *postLocOrig);
            memcpy(b, &postIdxGlobalRes, sizeof(postIdxGlobalRes));
            b += sizeof(postIdxGlobalRes);
            memcpy(b, postConn->get_wDataHead(arborId, idxExt), (size_t) blocksize[neighbor] * sizeof(pvwdata_t));
            b += blocksize[neighbor]*sizeof(pvwdata_t);
         }
      }
      assert(b==((char *) sendbuf[neighbor]) + buffersize[neighbor]);

      MPI_Isend(sendbuf[neighbor], buffersize[neighbor], MPI_CHAR, nbrIdx, icComm->getTag(neighbor), icComm->communicator(), &request[neighbor]);
   }
#endif // PV_USE_MPI

   const int nkRestrictedOrig = postConn->getPostNonextStrides()->sy; // preLocOrig->nx*preLocOrig->nf; // a stride in postConn
   int nPreRestrictedOrig = postConn->preSynapticLayer()->getNumNeurons();
   for (int kPreRestrictedOrig = 0; kPreRestrictedOrig < nPreRestrictedOrig; kPreRestrictedOrig++) {
      int kPreExtendedOrig = kIndexExtended(kPreRestrictedOrig, preLocOrig->nx, preLocOrig->ny, preLocOrig->nf, preLocOrig->halo.lt, preLocOrig->halo.rt, preLocOrig->halo.dn, preLocOrig->halo.up);
      PVPatch * patchOrig = postConn->getWeights(kPreExtendedOrig, arborId);
      int nk = patchOrig->nx * postConn->fPatchSize();
      int ny = patchOrig->ny;
      pvwdata_t * weightvaluesorig = postConn->get_wData(arborId, kPreExtendedOrig);
      int kPostRestrictedOrigBase = (int) postConn->getGSynPatchStart(kPreExtendedOrig, arborId);
      int kPostRestrictedTranspose = kPreRestrictedOrig;
      for (int y=0; y<ny; y++) {
         for (int k=0; k<nk; k++) {
            pvwdata_t w = weightvaluesorig[y*postConn->yPatchStride()+k];
            int kPostRestrictedOrig = kPostRestrictedOrigBase + y*nkRestrictedOrig+k;
            int kPreRestrictedTranspose = kPostRestrictedOrig;
            int kPreExtendedTranspose = kIndexExtended(kPreRestrictedTranspose, preLocTranspose->nx, preLocTranspose->ny, preLocTranspose->nf, preLocTranspose->halo.lt, preLocTranspose->halo.rt, preLocTranspose->halo.dn, preLocTranspose->halo.up);
            PVPatch * patchTranspose = getWeights(kPreExtendedTranspose, arborId);
            size_t gSynPatchStartTranspose = getGSynPatchStart(kPreExtendedTranspose, arborId);
            // Need to find which pixel in the patch is tied to kPostRestrictedTranspose
            // assert((size_t) kPostRestrictedTranspose>=gSynPatchStartTranspose);
            int moveFromOffset = kPostRestrictedTranspose-(int) gSynPatchStartTranspose;
            div_t coordsFromOffset = div(moveFromOffset, getPostNonextStrides()->sy);
            int yt = coordsFromOffset.quot;
            int kt = coordsFromOffset.rem;
            get_wData(arborId, kPreExtendedTranspose)[yt*syp+kt] = w;
         }
      }
   }

#ifdef PV_USE_MPI
   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      if (!hasRestrictedNeighbor[neighbor]) { continue; }
      int nbrIdx = icComm->neighborIndex(parent->columnId(), neighbor);
      assert(nbrIdx>=0); // If neighborIndex is negative, there is no neighbor in that direction so hasRestrictedNeighbor should be false

      MPI_Recv(recvbuf[neighbor], buffersize[neighbor], MPI_CHAR, nbrIdx, icComm->getReverseTag(neighbor), icComm->communicator(), MPI_STATUS_IGNORE);
      char * b = (char *) recvbuf[neighbor];
      int postGlobalResStrideY = postLocOrig->nxGlobal*postLocOrig->nf;
      for (int p=0; p<size[neighbor]; p++) {
         int origPreIndex;
         memcpy(&origPreIndex, b, sizeof(origPreIndex));
         b += sizeof(origPreIndex);
         PVPatch patch;
         memcpy(&patch, b, sizeof(patch));
         b += sizeof(patch);
         int postIdxGlobalRes;
         memcpy(&postIdxGlobalRes, b, sizeof(postIdxGlobalRes));
         b += sizeof(postIdxGlobalRes);
         for (int f=0; f<postConn->preSynapticLayer()->getLayerLoc()->nf; f++) {
            for (int y=0; y<patch.ny; y++) {
               for (int k=0; k<patch.nx*postConn->fPatchSize(); k++) {
                  int origPostIndex = postIdxGlobalRes + y*postGlobalResStrideY + k;
                  pvwdata_t w = ((pvwdata_t *) b)[patch.offset + y*postConn->yPatchStride()+k];

                  int transposePreGlobalRes = origPostIndex;
                  int transposePreGlobalXRes = kxPos(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
                  int transposePreLocalXRes = transposePreGlobalXRes - preLocTranspose->kx0;
                  int transposePreLocalXExt = transposePreLocalXRes + preLocTranspose->halo.lt;
                  assert(transposePreLocalXExt >= 0 && transposePreLocalXExt < preLocTranspose->nx+preLocTranspose->halo.lt+preLocTranspose->halo.rt);
                  int transposePreGlobalYRes = kyPos(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
                  int transposePreLocalYRes = transposePreGlobalYRes - preLocTranspose->ky0;
                  int transposePreLocalYExt = transposePreLocalYRes + preLocTranspose->halo.dn;
                  assert(transposePreLocalYExt >= 0 && transposePreLocalYExt < preLocTranspose->ny+preLocTranspose->halo.dn+preLocTranspose->halo.up);
                  int transposePreFeature = featureIndex(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
                  int transposePreLocalExt = kIndex(transposePreLocalXExt, transposePreLocalYExt, transposePreFeature, preLocTranspose->nx+preLocTranspose->halo.lt+preLocTranspose->halo.rt, preLocTranspose->ny+preLocTranspose->halo.dn+preLocTranspose->halo.up, preLocTranspose->nf);

                  int transposePostGlobalRes = origPreIndex;
                  int transposePostGlobalXRes = kxPos(transposePostGlobalRes, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
                  int origPreLocalXRes = transposePostGlobalXRes - preLocOrig->kx0;
                  assert(origPreLocalXRes>=0 && transposePostGlobalXRes<preLocOrig->nxGlobal);
                  int transposePostGlobalYRes = kyPos(transposePostGlobalRes, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
                  int origPreLocalYRes = transposePostGlobalYRes - preLocOrig->ky0;
                  assert(origPreLocalYRes>=0 && transposePostGlobalYRes<preLocOrig->nyGlobal);
                  int transposePostLocalRes = kIndex(origPreLocalXRes, origPreLocalYRes, f, postLocTranspose->nx, postLocTranspose->ny, postLocTranspose->nf);

                  PVPatch * transposePatch = getWeights(transposePreLocalExt, arborId);
                  int transposeGSynPatchStart = (int) getGSynPatchStart(transposePreLocalExt, arborId);
                  int transposeGSynOffset = transposePostLocalRes - transposeGSynPatchStart;
                  div_t coordsFromOffset = div(transposeGSynOffset, getPostNonextStrides()->sy);
                  int yt = coordsFromOffset.quot;
                  int kt = coordsFromOffset.rem;
                  get_wData(arborId, transposePreLocalExt)[yt*syp+kt] = w;
               }
            }
            b += sizeof(pvwdata_t)*postConn->xPatchSize()*postConn->yPatchSize()*postConn->fPatchSize();
         }
      }
      assert(b == ((char *) recvbuf[neighbor]) + buffersize[neighbor]);
   }

   // Free the receive buffers
   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      free(recvbuf[neighbor]); recvbuf[neighbor] = NULL;
   }

   // Free the send buffers.  Since a different process receives and it might be behind, need to call MPI_Test to see if the send was received.
   int numsent = 0;
   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
      if (request[neighbor]) {
         numsent++;
      }
   }
   while(numsent > 0) {
      for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
         if (request[neighbor]) {
            int flag = false;
            MPI_Test(&request[neighbor], &flag, MPI_STATUS_IGNORE);
            if (flag) {
               assert(request[neighbor] == MPI_REQUEST_NULL);
               request[neighbor] = NULL;
               numsent--;
               free(sendbuf[neighbor]); sendbuf[neighbor] = NULL;
            }
         }
      }
   }
#endif // PV_USE_MPI
   return PV_SUCCESS;
}

int privateTransposeConn::mpiexchangesize(int neighbor, int * size, int * startx, int * stopx, int * starty, int * stopy, int * blocksize, size_t * buffersize) {
   const PVLayerLoc * preLocOrig = postConn->preSynapticLayer()->getLayerLoc();
   PVHalo const * halo = &preLocOrig->halo;
   const int nx = preLocOrig->nx;
   const int ny = preLocOrig->ny;
   switch(neighbor) {
   case LOCAL:
      assert(0);
      break;
   case NORTHWEST:
      *startx = 0; *stopx = halo->lt;
      *starty = 0; *stopy = halo->up;
      break;
   case NORTH:
      *startx = halo->lt; *stopx = halo->lt + nx;
      *starty = 0; *stopy = halo->up;
      break;
   case NORTHEAST:
      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
      *starty = 0; *stopy = halo->up;
      break;
   case WEST:
      *startx = 0; *stopx = halo->lt;
      *starty = halo->up; *stopy = halo->up + ny;
      break;
   case EAST:
      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
      *starty = halo->up; *stopy = halo->up + ny;
      break;
   case SOUTHWEST:
      *startx = 0; *stopx = halo->lt;
      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
      break;
   case SOUTH:
      *startx = halo->lt; *stopx = halo->lt + nx;
      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
      break;
   case SOUTHEAST:
      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
      break;
   default:
      assert(0);
      break;
   }
   *size = (*stopx-*startx)*(*stopy-*starty);
   *blocksize = preLocOrig->nf * postConn->xPatchSize() * postConn->yPatchSize() * postConn->fPatchSize();
   // Each block is a contiguous set of preLocOrig->nf weight patches.  We also need to send the presynaptic index, the PVPatch geometry and the aPostOffset.
   // for each block.  This assumes that each of the preLocOrig->nf patches for a given (x,y) site has the same aPostOffset and PVPatch values.
   *buffersize = (size_t) *size * ( sizeof(int) + sizeof(PVPatch) + sizeof(int) + (size_t) *blocksize * sizeof(pvwdata_t));
   return PV_SUCCESS;
}

int privateTransposeConn::transposeSharedWeights(int arborId) {
   // compute the transpose of postConn->kernelPatches and
   // store into this->kernelPatches
   // assume scale factors are 1 and that nxp, nyp are odd.

   int xscalediff = pre->getXScale()-post->getXScale();
   int yscalediff = pre->getYScale()-post->getYScale();
   // scalediff>0 means privateTransposeConn's post--that is, the postConn's pre--has a higher neuron density

   int numFBKernelPatches = getNumDataPatches();
   int numFFKernelPatches = postConn->getNumDataPatches();

   if( xscalediff <= 0 && yscalediff <= 0) {
      int xscaleq = (int) pow(2,-xscalediff);
      int yscaleq = (int) pow(2,-yscalediff);

      int kerneloffsetx = 0;
      int kerneloffsety = 0;
      if(nxp%2 == 0){
         kerneloffsetx = xscaleq/2;
      }
      if(nyp%2 == 0){
         kerneloffsety = yscaleq/2;
      }

      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
         int nfFB = nfp;
         assert(numFFKernelPatches == nfFB);
         int nxFB = nxp; // kpFB->nx;
         int nyFB = nyp; // kpFB->ny;
         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                  int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                  int kernelnumberFF = kfFB;
                  // PVPatch * kpFF = postConn>getKernelPatch(0, kernelnumberFF);
                  pvwdata_t * dataStartFF = postConn->get_wDataHead(arborId, kernelnumberFF);
                  int nxpFF = postConn->xPatchSize();
                  int nypFF = postConn->yPatchSize();
                  assert(numFBKernelPatches == postConn->fPatchSize() * xscaleq * yscaleq);
                  int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, postConn->fPatchSize());

                  //Calculate x and y position of the FB kernel, with an offset for the case of even patches
                  int kxFFoffset = (kxPos(kernelnumberFB, xscaleq, yscaleq, postConn->fPatchSize()) + kerneloffsetx) % xscaleq;
                  int kxFF = (nxp - 1 - kxFB) * xscaleq + kxFFoffset;

                  int kyFFoffset = (kyPos(kernelnumberFB, xscaleq, yscaleq, postConn->fPatchSize()) + kerneloffsety) % yscaleq;
                  int kyFF = (nyp - 1 - kyFB) * yscaleq + kyFFoffset;

                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, postConn->fPatchSize());

                  // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
                  // kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
               }
            }
         }
      }
   }
   else if( xscalediff >= 0 && yscalediff >= 0) {
      int xscaleq = (int) pow(2,xscalediff);
      int yscaleq = (int) pow(2,yscalediff);

      int kerneloffsetx = 0;
      int kerneloffsety = 0;
      if((nxp/xscaleq)%2 == 0){
         kerneloffsetx = xscaleq/2;
      }
      if((nyp/yscaleq)%2 == 0){
         kerneloffsety = yscaleq/2;
      }

      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
         int nxFB = nxp; // kpFB->nx;
         int nyFB = nyp; // kpFB->ny;
         int nfFB = nfp;
         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
            int precelloffsety = (kyFB + kerneloffsety) % yscaleq;
            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
               int precelloffsetx = (kxFB + kerneloffsetx) % xscaleq;
               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                  int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                  pvwdata_t * dataStartFF = postConn->get_wDataHead(arborId, kernelnumberFF);
                  int nxpFF = postConn->xPatchSize();
                  int nypFF = postConn->yPatchSize();
                  int kxFF = (nxp-kxFB-1)/xscaleq;
                  assert(kxFF >= 0 && kxFF < postConn->xPatchSize());
                  int kyFF = (nyp-kyFB-1)/yscaleq;
                  assert(kyFF >= 0 && kyFF < postConn->yPatchSize());
                  int kfFF = kernelnumberFB;
                  assert(kfFF >= 0 && kfFF < postConn->fPatchSize());
                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, postConn->fPatchSize());
                  int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
               }
            }
         }
      }
   }
   else {
      fprintf(stderr,"xscalediff = %d, yscalediff = %d: the case of many-to-one in one dimension and one-to-many in the other"
            "has not yet been implemented.\n", xscalediff, yscalediff);
      exit(1);
   }

   return PV_SUCCESS;
}  // privateTransposeConn::transposeKernels()

int privateTransposeConn::reduceKernels(int arborID) {
   // Values are taken from postConn.  If postConn keeps kernels synchronized, then privateTransposeConn stays synchronized automatically.
   // If postConn does not, then privateTransposeConn shouldn't either.
   return PV_SUCCESS;
}

int privateTransposeConn::deliver() {
   //Sanity check, this should NEVER get called
   std::cout << "Fatal error, privateTransposeConn deliver got called\n";
   exit(-1);
}



} // end namespace PV
