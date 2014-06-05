/* CloneKernelConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneKernelConn.hpp"

namespace PV {

CloneKernelConn::CloneKernelConn(){
   initialize_base();
}

CloneKernelConn::CloneKernelConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int CloneKernelConn::initialize_base() {
   originalConn = NULL;
   return PV_SUCCESS;
}

int CloneKernelConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int CloneKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

// TODO: Merge with Sheng's changes for non-shared weight case
void CloneKernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void CloneKernelConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void CloneKernelConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", NULL);
   }
}

void CloneKernelConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int CloneKernelConn::setWeightInitializer() {
   weightInitializer = new InitCloneKernelWeights();
   return PV_SUCCESS;
}

int CloneKernelConn::constructWeights() {
   int status = PV_SUCCESS;

   // CloneKernelConn::ioParam_shrinkPatches does nothing; shrinkPatches_flag is set in communicateInitInfo()

   // if( status == PV_SUCCESS ) status = setPatchSize(NULL);
   if( status == PV_SUCCESS ) status = setPatchStrides();

   wPatches = this->originalConn->get_wPatches();
   wDataStart = this->originalConn->get_wDataStart();
   gSynPatchStart = this->originalConn->getGSynPatchStart();
   aPostOffset = this->originalConn->getAPostOffset();
   dwDataStart = this->originalConn->get_dwDataStart();

   // Don't call initPlasticityPatches since plasticityFlag is always false.
   // Don't call shrinkPatches() since the original connection will have already shrunk patches
   return status;
}

void CloneKernelConn::constructWeightsOutOfMemory() {
   connOutOfMemory("CloneKernelConn::constructWeightsOutOfMemory()");
}

int CloneKernelConn::createAxonalArbors(int arborId) {
   return PV_SUCCESS;
}

PVPatch *** CloneKernelConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart) {
   return patches;
   // nothing to be done as the weight patches point to originalConn's space.
}

// We override many read-methods because CloneKernelConn will use
// originalConn's values.  communicateInitInfo will check if the associated
// parameters exist in params for theCloneKernelConn group, and whether they
// are consistent with the originalConn parameters.
// If consistent, issue a warning that the param is unnecessary and continue.
// If inconsistent, issue an error and quit.
// We can't do that in the read-method because we can't be sure originalConn
// has set its own parameter yet (or even if it's been instantiated),
// and in theory originalConn could be a subclass that determines
// the parameter some way other than reading its own parameter
// group's param directly.

void CloneKernelConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   // During the communication phase, shrinkPatches_flag will be copied from originalConn
}

void CloneKernelConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void CloneKernelConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false; // CloneKernelConn updates automatically, since it's done using pointer magic.
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

void CloneKernelConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nxp will be copied from originalConn
}
void CloneKernelConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nyp will be copied from originalConn
}
void CloneKernelConn::ioParam_nxpShrunken(enum ParamsIOFlag ioFlag) {
   // CloneKernelConn does not use nxpShrunken
}
void CloneKernelConn::ioParam_nypShrunken(enum ParamsIOFlag ioFlag) {
   // CloneKernelConn does not use nypShrunken
}

void CloneKernelConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // During the communication phase, nfp will be copied from originalConn
}

int CloneKernelConn::communicateInitInfo() {
   // Need to set originalConn before calling HyPerConn::communicate, since HyPerConn::communicate calls setPatchSize, which needs originalConn.
   originalConn = parent->getConnFromName(originalConnName);
   if (originalConn == NULL) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: originalConnName \"%s\" is not a connection in the column.\n",
            name, parent->columnId(), originalConnName);
   }
   if (originalConn->usingSharedWeights() == false) {
      fprintf(stderr, "CloneKernelConn \"%s\" error in rank %d process: originalConnName \"%s\" must use shared weights.\n",
            name, parent->columnId(), originalConnName);
   }

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * classname = params->groupKeywordFromName(name);
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);
   status = HyPerConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   // Presynaptic layers of the CloneKernelConn and its original conn must have the same size, or the patches won't line up with each other.
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();

   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny || preLoc->nf != origPreLoc->nf ) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "%s \"%s\" error in rank %d process: CloneKernelConn and originalConn \"%s\" must have presynaptic layers with the same nx,ny,nf.\n",
               classname, name, parent->columnId(), originalConn->getName());
         fprintf(stderr, "{nx=%d, ny=%d, nf=%d} versus {nx=%d, ny=%d, nf=%d}\n",
                 preLoc->nx, preLoc->ny, preLoc->nf, origPreLoc->nx, origPreLoc->ny, origPreLoc->nf);
      }
      abort();
   }

   // Make sure the original's and the clone's margin widths stay equal
   originalConn->preSynapticLayer()->synchronizeMarginWidth(pre);
   pre->synchronizeMarginWidth(originalConn->preSynapticLayer());

   //Redudant read in case it's a clone of a clone
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   shrinkPatches_flag = originalConn->getShrinkPatches_flag();
   parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);

   return status;
}

int CloneKernelConn::setPatchSize() {
   assert(originalConn);
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   nxpShrunken = originalConn->getNxpShrunken();
   nypShrunken = originalConn->getNypShrunken();
   nfp = originalConn->fPatchSize();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nxpShrunken", nxpShrunken);
   parent->parameters()->handleUnnecessaryParameter(name, "nypShrunken", nypShrunken);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

int CloneKernelConn::updateState(double time, double dt) {
   update_timer->start();

   lastUpdateTime = originalConn->getLastUpdateTime();

   update_timer->stop();
   return PV_SUCCESS;
}

int CloneKernelConn::deleteWeights() {
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

CloneKernelConn::~CloneKernelConn() {
   free(originalConnName);
   deleteWeights();
}

} // end namespace PV
