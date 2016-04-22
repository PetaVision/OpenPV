/*
 * TransposeConn.cpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#include "TransposeConn.hpp"
#include "privateTransposeConn.hpp"

namespace PV {

TransposeConn::TransposeConn() {
   initialize_base();
}  // TransposeConn::~TransposeConn()

TransposeConn::TransposeConn(const char * name, HyPerCol * hc) {
   initialize_base();
   int status = initialize(name, hc);
}

TransposeConn::~TransposeConn() {
   free(originalConnName); originalConnName = NULL;
   deleteWeights();
   postConn = NULL;
   //Transpose conn doesn't allocate postToPreActivity
   postToPreActivity = NULL;
}  // TransposeConn::~TransposeConn()

int TransposeConn::initialize_base() {
   plasticityFlag = false; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params
   weightUpdateTime = 0;
   // TransposeConn::initialize_base() gets called after
   // HyPerConn::initialize_base() so these default values override
   // those in HyPerConn::initialize_base().
   // TransposeConn::initialize_base() gets called before
   // HyPerConn::initialize(), so these values still get overridden
   // by the params file values.

   originalConnName = NULL;
   originalConn = NULL;
   needFinalize = true;
   return PV_SUCCESS;
}  // TransposeConn::initialize_base()

int TransposeConn::initialize(const char * name, HyPerCol * hc) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) status = HyPerConn::initialize(name, hc, NULL, NULL);
   return status;
}

int TransposeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

// We override many ioParam-methods because TransposeConn will determine
// the associated parameters from the originalConn's values.
// communicateInitInfo will check if those parameters exist in params for
// the CloneKernelConn group, and whether they are consistent with the
// originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void TransposeConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "sharedWeights");
   }
}

void TransposeConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // TransposeConn doesn't use a weight initializer
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void TransposeConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // During the setInitialValues phase, the conn will be computed from the original conn, so initializeFromCheckpointFlag is not needed.
}

void TransposeConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
}

void TransposeConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void TransposeConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      // make sure that TransposePoolingConn always checks if its originalConn has updated
      triggerFlag = false;
      triggerLayerName = NULL;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL);
   }
}

void TransposeConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      combine_dW_with_W_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
}

void TransposeConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nxp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nyp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // TransposeConn determines nfp from originalConn, during communicateInitInfo
}

void TransposeConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      dWMax = 1.0;
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax", dWMax);
   }
}

void TransposeConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag);
   }
}

void TransposeConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
}

void TransposeConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      // Every timestep needUpdate checks originalConn's lastUpdateTime against transpose's lastUpdateTime, so weightUpdatePeriod and initialWeightUpdateTime aren't needed
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime", initialWeightUpdateTime);
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void TransposeConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      shrinkPatches_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   }
}

void TransposeConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      normalizer = NULL;
      normalizeMethod = strdup("none");
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none");
   }
}

void TransposeConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

#ifdef OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
InitWeights * TransposeConn::handleMissingInitWeights(PVParams * params) {
   // TransposeConn doesn't use InitWeights; it initializes the weight by transposing the initial weights of originalConn
   return NULL;
}
#endif // OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.

int TransposeConn::communicateInitInfo() {
   int status = PV_SUCCESS;
   BaseConnection * originalConnBase = parent->getConnFromName(this->originalConnName);
   if (originalConnBase==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalConnName \"%s\" does not refer to any connection in the column.\n", this->getKeyword(), name, this->originalConnName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   this->originalConn = dynamic_cast<HyPerConn *>(originalConnBase);
   if (originalConn == NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: originalConnName \"%s\" is not an existing connection.\n", name, originalConnName);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) return status;

   if (!originalConn->getInitInfoCommunicatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = this->getKeyword();
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its communicateInitInfo stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   sharedWeights = originalConn->usingSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);

   //plasticityFlag = originalConn->getPlasticityFlag();
   //parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);

   if(originalConn->getShrinkPatches_flag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: original conn \"%s\" has shrinkPatches set to true.  TransposeConn has not been implemented for that case.\n", name, originalConn->getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   status = HyPerConn::communicateInitInfo(); // calls setPatchSize()
   if (status != PV_SUCCESS) return status;

   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPostLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny || preLoc->nf != origPostLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's pre layer and original connection's post layer must have the same dimensions.\n", this->getKeyword(), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", preLoc->nx, preLoc->ny, preLoc->nf, origPostLoc->nx, origPostLoc->ny, origPostLoc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc * postLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->postSynapticLayer()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny || postLoc->nf != origPreLoc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: transpose's post layer and original connection's pre layer must have the same dimensions.\n", this->getKeyword(), name);
         fprintf(stderr, "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n", postLoc->nx, postLoc->ny, postLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   originalConn->setNeedPost(true);

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if((updateGSynFromPostPerspective && receiveGpu) || allocPostDeviceWeights){
      originalConn->setAllocDeviceWeights();
   }
   if((!updateGSynFromPostPerspective && receiveGpu) || allocDeviceWeights){
      originalConn->setAllocPostDeviceWeights();
   }
#endif
   

   //Synchronize margines of this post and orig pre, and vice versa
   originalConn->preSynapticLayer()->synchronizeMarginWidth(post);
   post->synchronizeMarginWidth(originalConn->preSynapticLayer());

   originalConn->postSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->postSynapticLayer());


   return status;
}

int TransposeConn::setPatchSize() {
   // If originalConn is many-to-one, the transpose connection is one-to-many; then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.

   // Some of the code duplication might be eliminated by adding some functions to convert.h

   assert(pre && post);
   assert(originalConn);

   int xscaleDiff = pre->getXScale() - post->getXScale();
   int nxp_orig = originalConn->xPatchSize();
   int nyp_orig = originalConn->yPatchSize();
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
   // post->getLayerLoc()->nf must be the same as originalConn->preSynapticLayer()->getLayerLoc()->nf.
   // This requirement is checked in communicateInitInfo

   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//Device buffers live in origConn
int TransposeConn::allocateDeviceWeights(){
   return PV_SUCCESS;
}
int TransposeConn::allocatePostDeviceWeights(){
   return PV_SUCCESS;
}
#endif

//Set this post to orig
int TransposeConn::allocatePostConn(){
   std::cout << "Connection " << name << " setting " << originalConn->getName() << " as postConn\n";
   postConn = originalConn;
   //originalConn->postConn->allocatePostToPreBuffer();
   //postToPreActivity = originalConn->postConn->getPostToPreActivity();
   return PV_SUCCESS;
}

int TransposeConn::allocateDataStructures() {
   if (!originalConn->getDataStructuresAllocatedFlag()) {
      if (parent->columnId()==0) {
         const char * connectiontype = this->getKeyword();
         printf("%s \"%s\" must wait until original connection \"%s\" has finished its allocateDataStructures stage.\n", connectiontype, name, originalConn->getName());
      }
      return PV_POSTPONE;
   }

   int status = HyPerConn::allocateDataStructures();
   if (status != PV_SUCCESS) { return status; }

   normalizer = NULL;
   
   // normalize_flag = false; // replaced by testing whether normalizer!=NULL
   return status;
}

int TransposeConn::constructWeights(){
   setPatchStrides();
   wPatches = this->originalConn->postConn->get_wPatches();
   wDataStart = this->originalConn->postConn->get_wDataStart();
   gSynPatchStart = this->originalConn->postConn->getGSynPatchStart();
   aPostOffset = this->originalConn->postConn->getAPostOffset();
   dwDataStart = this->originalConn->postConn->get_dwDataStart();
   return PV_SUCCESS;
}

int TransposeConn::deleteWeights() {
   // Have to make sure not to free memory belonging to originalConn.
   // Set pointers that point into originalConn to NULL so that free() has no effect
   // when HyPerConn::deleteWeights or HyPerConn::deleteWeights is called
	   wPatches = NULL;
	   wDataStart = NULL;
	   gSynPatchStart = NULL;
	   aPostOffset = NULL;
	   dwDataStart = NULL;
//   for(int arbor=0; arbor<numberOfAxonalArborLists(); arbor++) {
//      get_wPatches()[arbor] = NULL;
//      set_wDataStart(arbor,NULL);
//   }
   // set_kernelPatches(NULL);

   return 0; // HyPerConn::deleteWeights(); // HyPerConn destructor calls HyPerConn::deleteWeights()
}

int TransposeConn::setInitialValues() {
   int status = PV_SUCCESS;
   if (originalConn->getInitialValuesSetFlag()) {
      status = HyPerConn::setInitialValues(); // calls initializeWeights
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

PVPatch*** TransposeConn::initializeWeights(PVPatch*** patches, pvwdata_t** dataStart) {
   // TransposeConn must wait until after originalConn has been normalized, so weight initialization doesn't take place until HyPerCol::run calls finalizeUpdate
   return patches;
}

bool TransposeConn::needUpdate(double timed, double dt) {
   return false;
}

int TransposeConn::updateState(double time, double dt) {
   return PV_SUCCESS;
}

double TransposeConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   return weightUpdateTime; // TransposeConn does not use weightUpdateTime to determine when to update
}

int TransposeConn::finalizeUpdate(double timed, double dt){
   //Orig conn is in charge of calling finalizeUpdate for postConn.
   return PV_SUCCESS;
}

//int TransposeConn::finalizeUpdate(double time, double dt) {
//   if (!needFinalize) { return PV_SUCCESS; }
//   int status = HyPerConn::finalizeUpdate(time, dt);
//   update_timer->start();
//   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
//      status = transpose(arborId);  // Apply changes in weights
//      if (status==PV_BREAK) { break; }
//      assert(status==PV_SUCCESS);
//   }
//
//   update_timer->stop();
//   return status;
//}
//
//int TransposeConn::transpose(int arborId) {
//   return sharedWeights ? transposeSharedWeights(arborId) : transposeNonsharedWeights(arborId);
//}
//
//int TransposeConn::transposeNonsharedWeights(int arborId) {
//   assert(usingSharedWeights()==false);
//   const PVLayerLoc * preLocOrig = originalConn->preSynapticLayer()->getLayerLoc();
//   const PVLayerLoc * postLocOrig = originalConn->postSynapticLayer()->getLayerLoc();
//   const PVLayerLoc * preLocTranspose = preSynapticLayer()->getLayerLoc();
//   const PVLayerLoc * postLocTranspose = postSynapticLayer()->getLayerLoc();
//#ifdef PV_USE_MPI
//   InterColComm * icComm = parent->icCommunicator();
//   pvwdata_t * sendbuf[NUM_NEIGHBORHOOD];
//   pvwdata_t * recvbuf[NUM_NEIGHBORHOOD];
//   int size[NUM_NEIGHBORHOOD];
//   int startx[NUM_NEIGHBORHOOD];
//   int starty[NUM_NEIGHBORHOOD];
//   int stopx[NUM_NEIGHBORHOOD];
//   int stopy[NUM_NEIGHBORHOOD];
//   int blocksize[NUM_NEIGHBORHOOD];
//   MPI_Request request[NUM_NEIGHBORHOOD];
//   size_t buffersize[NUM_NEIGHBORHOOD];
//   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//      if (neighbor==LOCAL || icComm->neighborIndex(parent->columnId(), neighbor)<0 ) {
//         size[neighbor] = 0;
//         startx[neighbor] = -1;
//         starty[neighbor] = -1;
//         stopx[neighbor] = -1;
//         stopy[neighbor] = -1;
//         blocksize[neighbor] = 0;
//         buffersize[neighbor] = (size_t) 0;
//         sendbuf[neighbor] = NULL;
//         recvbuf[neighbor] = NULL;
//         request[neighbor] = NULL;
//      }
//      else {
//         mpiexchangesize(neighbor,  &size[neighbor], &startx[neighbor], &stopx[neighbor], &starty[neighbor], &stopy[neighbor], &blocksize[neighbor], &buffersize[neighbor]);
//         sendbuf[neighbor] = (pvwdata_t *) malloc(buffersize[neighbor]);
//         if (sendbuf[neighbor]==NULL) {
//            fprintf(stderr, "%s \"%s\": Rank %d process unable to allocate memory for Transpose send buffer: %s\n", this->getKeyword(), name, parent->columnId(), strerror(errno));
//            exit(EXIT_FAILURE);
//         }
//         recvbuf[neighbor] = (pvwdata_t *) malloc(buffersize[neighbor]);
//         if (recvbuf[neighbor]==NULL) {
//            fprintf(stderr, "%s \"%s\": Rank %d process unable to allocate memory for Transpose receive buffer: %s\n", this->getKeyword(), name, parent->columnId(), strerror(errno));
//            exit(EXIT_FAILURE);
//         }
//         request[neighbor] = NULL;
//      }
//   }
//
//   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//      if (neighbor==LOCAL) { continue; }
//      int nbrIdx = icComm->neighborIndex(parent->columnId(), neighbor);
//      if (nbrIdx<0) { continue; }
//      if (icComm->reverseDirection(parent->columnId(), neighbor) + neighbor != NUM_NEIGHBORHOOD) { continue; }
//
//      char * b = (char *) sendbuf[neighbor];
//      for (int y=starty[neighbor]; y<stopy[neighbor]; y++) {
//         for (int x=startx[neighbor]; x<stopx[neighbor]; x++) {
//            int idxExt = kIndex(x,y,0,preLocOrig->nx+preLocOrig->halo.lt+preLocOrig->halo.rt,preLocOrig->ny+preLocOrig->halo.dn+preLocOrig->halo.up,preLocOrig->nf);
//            int xGlobalExt = x+preLocOrig->kx0;
//            int xGlobalRes = xGlobalExt-preLocOrig->halo.lt;
//            int yGlobalExt = y+preLocOrig->ky0;
//            int yGlobalRes = yGlobalExt-preLocOrig->halo.up;
//            if (xGlobalRes >= preLocOrig->kx0 && xGlobalRes < preLocOrig->kx0+preLocOrig->nx && yGlobalRes >= preLocOrig->ky0 && yGlobalRes < preLocOrig->ky0+preLocOrig->ny) {
//               fprintf(stderr, "Rank %d, connection \"%s\", x=%d, y=%d, neighbor=%d: xGlobalRes = %d, preLocOrig->kx0 = %d, preLocOrig->nx = %d\n", parent->columnId(), name, x, y, neighbor, xGlobalRes, preLocOrig->kx0, preLocOrig->nx);
//               fprintf(stderr, "Rank %d, connection \"%s\", x=%d, y=%d, neighbor=%d: yGlobalRes = %d, preLocOrig->ky0 = %d, preLocOrig->ny = %d\n", parent->columnId(), name, x, y, neighbor, yGlobalRes, preLocOrig->ky0, preLocOrig->ny);
//               exit(EXIT_FAILURE);
//            }
//            int idxGlobalRes = kIndex(xGlobalRes, yGlobalRes, 0, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
//            memcpy(b, &idxGlobalRes, sizeof(idxGlobalRes));
//            b += sizeof(idxGlobalRes);
//            PVPatch * patchOrig = originalConn->getWeights(idxExt, arborId);
//            memcpy(b, patchOrig, sizeof(*patchOrig));
//            b += sizeof(*patchOrig);
//            int postIdxRes = (int) originalConn->getGSynPatchStart(idxExt, arborId);
//            int postIdxExt = kIndexExtended(postIdxRes, postLocOrig->nx, postLocOrig->ny, postLocOrig->nf, postLocOrig->halo.lt, postLocOrig->halo.rt, postLocOrig->halo.dn, postLocOrig->halo.up);
//            int postIdxGlobalRes = globalIndexFromLocal(postIdxRes, *postLocOrig);
//            memcpy(b, &postIdxGlobalRes, sizeof(postIdxGlobalRes));
//            b += sizeof(postIdxGlobalRes);
//            memcpy(b, originalConn->get_wDataHead(arborId, idxExt), (size_t) blocksize[neighbor] * sizeof(pvwdata_t));
//            b += blocksize[neighbor]*sizeof(pvwdata_t);
//         }
//      }
//      assert(b==((char *) sendbuf[neighbor]) + buffersize[neighbor]);
//
//      MPI_Isend(sendbuf[neighbor], buffersize[neighbor], MPI_CHAR, nbrIdx, icComm->getTag(neighbor), icComm->communicator(), &request[neighbor]);
//   }
//#endif // PV_USE_MPI
//
//   const int nkRestrictedOrig = originalConn->getPostNonextStrides()->sy; // preLocOrig->nx*preLocOrig->nf; // a stride in originalConn
//   int nPreRestrictedOrig = originalConn->preSynapticLayer()->getNumNeurons();
//   for (int kPreRestrictedOrig = 0; kPreRestrictedOrig < nPreRestrictedOrig; kPreRestrictedOrig++) {
//      int kPreExtendedOrig = kIndexExtended(kPreRestrictedOrig, preLocOrig->nx, preLocOrig->ny, preLocOrig->nf, preLocOrig->halo.lt, preLocOrig->halo.rt, preLocOrig->halo.dn, preLocOrig->halo.up);
//      PVPatch * patchOrig = originalConn->getWeights(kPreExtendedOrig, arborId);
//      int nk = patchOrig->nx * originalConn->fPatchSize();
//      int ny = patchOrig->ny;
//      pvwdata_t * weightvaluesorig = originalConn->get_wData(arborId, kPreExtendedOrig);
//      int kPostRestrictedOrigBase = (int) originalConn->getGSynPatchStart(kPreExtendedOrig, arborId);
//      int kPostRestrictedTranspose = kPreRestrictedOrig;
//      for (int y=0; y<ny; y++) {
//         for (int k=0; k<nk; k++) {
//            pvwdata_t w = weightvaluesorig[y*originalConn->yPatchStride()+k];
//            int kPostRestrictedOrig = kPostRestrictedOrigBase + y*nkRestrictedOrig+k;
//            int kPreRestrictedTranspose = kPostRestrictedOrig;
//            int kPreExtendedTranspose = kIndexExtended(kPreRestrictedTranspose, preLocTranspose->nx, preLocTranspose->ny, preLocTranspose->nf, preLocTranspose->halo.lt, preLocTranspose->halo.rt, preLocTranspose->halo.dn, preLocTranspose->halo.up);
//            PVPatch * patchTranspose = getWeights(kPreExtendedTranspose, arborId);
//            size_t gSynPatchStartTranspose = getGSynPatchStart(kPreExtendedTranspose, arborId);
//            // Need to find which pixel in the patch is tied to kPostRestrictedTranspose
//            // assert((size_t) kPostRestrictedTranspose>=gSynPatchStartTranspose);
//            int moveFromOffset = kPostRestrictedTranspose-(int) gSynPatchStartTranspose;
//            div_t coordsFromOffset = div(moveFromOffset, getPostNonextStrides()->sy);
//            int yt = coordsFromOffset.quot;
//            int kt = coordsFromOffset.rem;
//            get_wData(arborId, kPreExtendedTranspose)[yt*syp+kt] = w;
//         }
//      }
//   }
//
//#ifdef PV_USE_MPI
//   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//      if (neighbor==LOCAL) { continue; }
//      int nbrIdx = icComm->neighborIndex(parent->columnId(), neighbor);
//      if (nbrIdx<0) { continue; }
//      if (icComm->reverseDirection(parent->columnId(), neighbor) + neighbor != NUM_NEIGHBORHOOD) { continue; }
//
//      MPI_Recv(recvbuf[neighbor], buffersize[neighbor], MPI_CHAR, nbrIdx, icComm->getReverseTag(neighbor), icComm->communicator(), MPI_STATUS_IGNORE);
//      char * b = (char *) recvbuf[neighbor];
//      int postGlobalResStrideY = postLocOrig->nxGlobal*postLocOrig->nf;
//      for (int p=0; p<size[neighbor]; p++) {
//         int origPreIndex;
//         memcpy(&origPreIndex, b, sizeof(origPreIndex));
//         b += sizeof(origPreIndex);
//         PVPatch patch;
//         memcpy(&patch, b, sizeof(patch));
//         b += sizeof(patch);
//         int postIdxGlobalRes;
//         memcpy(&postIdxGlobalRes, b, sizeof(postIdxGlobalRes));
//         b += sizeof(postIdxGlobalRes);
//         for (int f=0; f<originalConn->preSynapticLayer()->getLayerLoc()->nf; f++) {
//            for (int y=0; y<patch.ny; y++) {
//               for (int k=0; k<patch.nx*originalConn->fPatchSize(); k++) {
//                  int origPostIndex = postIdxGlobalRes + y*postGlobalResStrideY + k;
//                  pvwdata_t w = ((pvwdata_t *) b)[patch.offset + y*originalConn->yPatchStride()+k];
//
//                  int transposePreGlobalRes = origPostIndex;
//                  int transposePreGlobalXRes = kxPos(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
//                  int transposePreLocalXRes = transposePreGlobalXRes - preLocTranspose->kx0;
//                  int transposePreLocalXExt = transposePreLocalXRes + preLocTranspose->halo.lt;
//                  assert(transposePreLocalXExt >= 0 && transposePreLocalXExt < preLocTranspose->nx+preLocTranspose->halo.lt+preLocTranspose->halo.rt);
//                  int transposePreGlobalYRes = kyPos(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
//                  int transposePreLocalYRes = transposePreGlobalYRes - preLocTranspose->ky0;
//                  int transposePreLocalYExt = transposePreLocalYRes + preLocTranspose->halo.dn;
//                  assert(transposePreLocalYExt >= 0 && transposePreLocalYExt < preLocTranspose->ny+preLocTranspose->halo.dn+preLocTranspose->halo.up);
//                  int transposePreFeature = featureIndex(transposePreGlobalRes, preLocTranspose->nxGlobal, preLocTranspose->nyGlobal, preLocTranspose->nf);
//                  int transposePreLocalExt = kIndex(transposePreLocalXExt, transposePreLocalYExt, transposePreFeature, preLocTranspose->nx+preLocTranspose->halo.lt+preLocTranspose->halo.rt, preLocTranspose->ny+preLocTranspose->halo.dn+preLocTranspose->halo.up, preLocTranspose->nf);
//
//                  int transposePostGlobalRes = origPreIndex;
//                  int transposePostGlobalXRes = kxPos(transposePostGlobalRes, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
//                  int origPreLocalXRes = transposePostGlobalXRes - preLocOrig->kx0;
//                  assert(origPreLocalXRes>=0 && transposePostGlobalXRes<preLocOrig->nxGlobal);
//                  int transposePostGlobalYRes = kyPos(transposePostGlobalRes, preLocOrig->nxGlobal, preLocOrig->nyGlobal, preLocOrig->nf);
//                  int origPreLocalYRes = transposePostGlobalYRes - preLocOrig->ky0;
//                  assert(origPreLocalYRes>=0 && transposePostGlobalYRes<preLocOrig->nyGlobal);
//                  int transposePostLocalRes = kIndex(origPreLocalXRes, origPreLocalYRes, f, postLocTranspose->nx, postLocTranspose->ny, postLocTranspose->nf);
//
//                  PVPatch * transposePatch = getWeights(transposePreLocalExt, arborId);
//                  int transposeGSynPatchStart = (int) getGSynPatchStart(transposePreLocalExt, arborId);
//                  int transposeGSynOffset = transposePostLocalRes - transposeGSynPatchStart;
//                  div_t coordsFromOffset = div(transposeGSynOffset, getPostNonextStrides()->sy);
//                  int yt = coordsFromOffset.quot;
//                  int kt = coordsFromOffset.rem;
//                  get_wData(arborId, transposePreLocalExt)[yt*syp+kt] = w;
//               }
//            }
//            b += sizeof(pvwdata_t)*originalConn->xPatchSize()*originalConn->yPatchSize()*originalConn->fPatchSize();
//         }
//      }
//      assert(b == ((char *) recvbuf[neighbor]) + buffersize[neighbor]);
//   }
//
//   // Free the receive buffers
//   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//      free(recvbuf[neighbor]); recvbuf[neighbor] = NULL;
//   }
//
//   // Free the send buffers.  Since a different process receives and it might be behind, need to
//   int numsent = 0;
//   for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//      if (request[neighbor]) {
//         numsent++;
//      }
//   }
//   while(numsent > 0) {
//      for (int neighbor=0; neighbor<NUM_NEIGHBORHOOD; neighbor++) {
//         if (request[neighbor]) {
//            int flag = false;
//            MPI_Test(&request[neighbor], &flag, MPI_STATUS_IGNORE);
//            if (flag) {
//               assert(request[neighbor] == MPI_REQUEST_NULL);
//               request[neighbor] = NULL;
//               numsent--;
//               free(sendbuf[neighbor]); sendbuf[neighbor] = NULL;
//            }
//         }
//      }
//   }
//#endif // PV_USE_MPI
//   return PV_SUCCESS;
//}
//
//int TransposeConn::mpiexchangesize(int neighbor, int * size, int * startx, int * stopx, int * starty, int * stopy, int * blocksize, size_t * buffersize) {
//   const PVLayerLoc * preLocOrig = originalConn->preSynapticLayer()->getLayerLoc();
//   PVHalo const * halo = &preLocOrig->halo;
//   const int nx = preLocOrig->nx;
//   const int ny = preLocOrig->ny;
//   switch(neighbor) {
//   case LOCAL:
//      assert(0);
//      break;
//   case NORTHWEST:
//      *startx = 0; *stopx = halo->lt;
//      *starty = 0; *stopy = halo->up;
//      break;
//   case NORTH:
//      *startx = halo->lt; *stopx = halo->lt + nx;
//      *starty = 0; *stopy = halo->up;
//      break;
//   case NORTHEAST:
//      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
//      *starty = 0; *stopy = halo->up;
//      break;
//   case WEST:
//      *startx = 0; *stopx = halo->lt;
//      *starty = halo->up; *stopy = halo->up + ny;
//      break;
//   case EAST:
//      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
//      *starty = halo->up; *stopy = halo->up + ny;
//      break;
//   case SOUTHWEST:
//      *startx = 0; *stopx = halo->lt;
//      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
//      break;
//   case SOUTH:
//      *startx = halo->lt; *stopx = halo->lt + nx;
//      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
//      break;
//   case SOUTHEAST:
//      *startx = nx + halo->lt; *stopx = nx + halo->lt + halo->rt;
//      *starty = ny + halo->up; *stopy = ny + halo->dn + halo->up;
//      break;
//   default:
//      assert(0);
//      break;
//   }
//   *size = (*stopx-*startx)*(*stopy-*starty);
//   *blocksize = preLocOrig->nf * originalConn->xPatchSize() * originalConn->yPatchSize() * originalConn->fPatchSize();
//   // Each block is a contiguous set of preLocOrig->nf weight patches.  We also need to send the presynaptic index, the PVPatch geometry and the aPostOffset.
//   // for each block.  This assumes that each of the preLocOrig->nf patches for a given (x,y) site has the same aPostOffset and PVPatch values.
//   *buffersize = (size_t) *size * ( sizeof(int) + sizeof(PVPatch) + sizeof(int) + (size_t) *blocksize * sizeof(pvwdata_t));
//   return PV_SUCCESS;
//}
//
//int TransposeConn::transposeSharedWeights(int arborId) {
//   // compute the transpose of originalConn->kernelPatches and
//   // store into this->kernelPatches
//   // assume scale factors are 1 and that nxp, nyp are odd.
//
//   int xscalediff = pre->getXScale()-post->getXScale();
//   int yscalediff = pre->getYScale()-post->getYScale();
//   // scalediff>0 means TransposeConn's post--that is, the originalConn's pre--has a higher neuron density
//
//   int numFBKernelPatches = getNumDataPatches();
//   int numFFKernelPatches = originalConn->getNumDataPatches();
//
//   if( xscalediff <= 0 && yscalediff <= 0) {
//      int xscaleq = (int) pow(2,-xscalediff);
//      int yscaleq = (int) pow(2,-yscalediff);
//
//
//      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
//         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
//         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
//         int nfFB = nfp;
//         assert(numFFKernelPatches == nfFB);
//         int nxFB = nxp; // kpFB->nx;
//         int nyFB = nyp; // kpFB->ny;
//         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
//            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
//               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
//                  int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
//                  int kernelnumberFF = kfFB;
//                  // PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
//                  pvwdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
//                  int nxpFF = originalConn->xPatchSize();
//                  int nypFF = originalConn->yPatchSize();
//                  assert(numFBKernelPatches == originalConn->fPatchSize() * xscaleq * yscaleq);
//                  int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
//                  int kxFFoffset = kxPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
//                  int kxFF = (nxp - 1 - kxFB) * xscaleq + kxFFoffset;
//                  int kyFFoffset = kyPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
//                  int kyFF = (nyp - 1 - kyFB) * yscaleq + kyFFoffset;
//                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
//                  // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
//                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
//                  // kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
//               }
//            }
//         }
//      }
//   }
//   else if( xscalediff > 0 && yscalediff > 0) {
//      int xscaleq = (int) pow(2,xscalediff);
//      int yscaleq = (int) pow(2,yscalediff);
//      for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
//         // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
//         pvwdata_t * dataStartFB = get_wDataHead(arborId, kernelnumberFB);
//         int nxFB = nxp; // kpFB->nx;
//         int nyFB = nyp; // kpFB->ny;
//         int nfFB = nfp;
//         for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
//            int precelloffsety = kyFB % yscaleq;
//            for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
//               int precelloffsetx = kxFB % xscaleq;
//               for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
//                  int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
//                  pvwdata_t * dataStartFF = originalConn->get_wDataHead(arborId, kernelnumberFF);
//                  int nxpFF = originalConn->xPatchSize();
//                  int nypFF = originalConn->yPatchSize();
//                  int kxFF = (nxp-kxFB-1)/xscaleq;
//                  assert(kxFF >= 0 && kxFF < originalConn->xPatchSize());
//                  int kyFF = (nyp-kyFB-1)/yscaleq;
//                  assert(kyFF >= 0 && kyFF < originalConn->yPatchSize());
//                  int kfFF = kernelnumberFB;
//                  assert(kfFF >= 0 && kfFF < originalConn->fPatchSize());
//                  int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
//                  int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
//                  dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
//               }
//            }
//         }
//      }
//   }
//   else {
//      fprintf(stderr,"xscalediff = %d, yscalediff = %d: the case of many-to-one in one dimension and one-to-many in the other"
//            "has not yet been implemented.\n", xscalediff, yscalediff);
//      exit(1);
//   }
//
//   return PV_SUCCESS;
//}  // TransposeConn::transposeKernels()
//
//int TransposeConn::reduceKernels(int arborID) {
//   // Values are taken from originalConn.  If originalConn keeps kernels synchronized, then TransposeConn stays synchronized automatically.
//   // If originalConn does not, then TransposeConn shouldn't either.
//   return PV_SUCCESS;
//}

BaseObject * createTransposeConn(char const * name, HyPerCol * hc) {
   return hc ? new TransposeConn(name, hc) : NULL;
}

} // end namespace PV
