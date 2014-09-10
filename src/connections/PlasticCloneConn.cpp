/* PlasticCloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "PlasticCloneConn.hpp"

namespace PV {

PlasticCloneConn::PlasticCloneConn(){
   initialize_base();
}

PlasticCloneConn::PlasticCloneConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int PlasticCloneConn::initialize_base() {
   return PV_SUCCESS;
}

int PlasticCloneConn::initialize(const char * name, HyPerCol * hc) {
   int status = CloneConn::initialize(name, hc);
   return status;
}

int PlasticCloneConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneConn::ioParamsFillGroup(ioFlag);
   return status;
}

// We override many read-methods because PlasticCloneConn will use
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


void PlasticCloneConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized");
   }
   // During the communication phase, shrinkPatches_flag will be copied from originalConn
}

void PlasticCloneConn::ioParam_useWindowPost(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "useWindowPost");
   }
   // During the communication phase, shrinkPatches_flag will be copied from originalConn
}

int PlasticCloneConn::communicateInitInfo() {
   // Need to set originalConn before calling HyPerConn::communicate, since HyPerConn::communicate calls setPatchSize, which needs originalConn.
   originalConn = parent->getConnFromName(originalConnName);
   if (originalConn == NULL) {
      fprintf(stderr, "PlasticCloneConn \"%s\" error in rank %d process: originalConnName \"%s\" is not a connection in the column.\n",
            name, parent->columnId(), originalConnName);
   }

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   const char * classname = params->groupKeywordFromName(name);
   sharedWeights = originalConn->usingSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);
   useWindowPost = originalConn->getUseWindowPost();
   parent->parameters()->handleUnnecessaryParameter(name, "useWindowPost", useWindowPost);
#ifdef PV_USE_MPI
   keepKernelsSynchronized_flag = originalConn->getKeepKernelsSynchronized();
   parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag);
#endif

   //Set plasticity flag to true for allocate
   plasticityFlag = true;
   //Grab dwMax for calculation of weights
   dWMax = originalConn->getDWMax();
   weightUpdatePeriod = originalConn->getWeightUpdatePeriod();

   status = HyPerConn::communicateInitInfo();
   if (status != PV_SUCCESS) return status;

   // Presynaptic layers of the PlasticCloneConn and its original conn must have the same size, or the patches won't line up with each other.
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * origPreLoc = originalConn->preSynapticLayer()->getLayerLoc();

   if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny || preLoc->nf != origPreLoc->nf ) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "%s \"%s\" error in rank %d process: PlasticCloneConn and originalConn \"%s\" must have presynaptic layers with the same nx,ny,nf.\n",
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

   //Add this clone to original conn
   originalConn->addClone(this);

   return status;
}

} // end namespace PV
