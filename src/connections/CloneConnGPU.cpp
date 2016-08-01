#include "../io/PVParams.hpp"
#include "CloneConnGPU.hpp"


namespace GPULCA {

CloneConnGPU::CloneConnGPU() {}

	CloneConnGPU::CloneConnGPU(const char* name, PV::HyPerCol* hc)
    : HyPerConnGPU(name, hc) {}

CloneConnGPU::~CloneConnGPU() {}

int CloneConnGPU::communicateInitInfo() {
  // Need to set originalConn before calling HyPerConn::communicate, since
  // HyPerConn::communicate calls setPatchSize, which needs originalConn.
  BaseConnection* originalConnBase = parent->getConnFromName(originalConnName);
  if (originalConnBase == NULL) {
    if (parent->columnId() == 0) {
      pvErrorNoExit().printf(
          "%s: originalConnName \"%s\" is not a connection in the column.\n",
          getDescription_c(), originalConnName);
    }
    MPI_Barrier(parent->getCommunicator()->communicator());
    exit(EXIT_FAILURE);
  }
  originalConn = dynamic_cast<HyPerConnGPU*>(originalConnBase);
  if (originalConn == NULL) {
    if (parent->columnId() == 0) {
      pvErrorNoExit().printf(
          "%s: originalConnName \"%s\" is not a HyPerConn or HyPerConn-derived "
          "class.\n",
          getDescription_c(), originalConnName);
    }
  }
  if (!originalConn->getInitInfoCommunicatedFlag()) {
    if (parent->columnId() == 0) {
      pvInfo().printf(
          "%s must wait until original connection \"%s\" has finished its "
          "communicateInitInfo stage.\n",
          getDescription_c(), originalConn->getName());
    }
    return PV_POSTPONE;
  }

  // Copy some parameters from originalConn.  Check if parameters exist is
  // the clone's param group, and issue a warning (if the param has the right
  // value) or an error (if it has the wrong value).
  int status = cloneParameters();

  status = HyPerConn::communicateInitInfo();
  if (status != PV_SUCCESS) return status;

  // Presynaptic layers of the CloneConn and its original conn must have the
  // same size, or the patches won't line up with each other.
  const PVLayerLoc* preLoc = pre->getLayerLoc();
  const PVLayerLoc* origPreLoc =
      originalConn->preSynapticLayer()->getLayerLoc();

  if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny ||
      preLoc->nf != origPreLoc->nf) {
    if (parent->getCommunicator()->commRank() == 0) {
      pvErrorNoExit(errorMessage);
      errorMessage.printf(
          "%s: CloneConn and originalConn \"%s\" must have presynaptic layers "
          "with the same nx,ny,nf.\n",
          getDescription_c(), parent->columnId(), originalConn->getName());
      errorMessage.printf(
          "{nx=%d, ny=%d, nf=%d} versus {nx=%d, ny=%d, nf=%d}\n", preLoc->nx,
          preLoc->ny, preLoc->nf, origPreLoc->nx, origPreLoc->ny,
          origPreLoc->nf);
    }
    MPI_Barrier(parent->getCommunicator()->communicator());
    abort();
  }

  // Make sure the original's and the clone's margin widths stay equal
  originalConn->preSynapticLayer()->synchronizeMarginWidth(pre);
  pre->synchronizeMarginWidth(originalConn->preSynapticLayer());

  // Make sure the original's and the clone's margin widths stay equal
  // Only if this layer receives from post for patch to data LUT
  if (getUpdateGSynFromPostPerspective()) {
    originalConn->postSynapticLayer()->synchronizeMarginWidth(post);
    post->synchronizeMarginWidth(originalConn->postSynapticLayer());
  }
  // Redudant read in case it's a clone of a clone

  return status;
}

int CloneConnGPU::cloneParameters() {
  // Copy sharedWeights, numAxonalArborLists, shrinkPatches_flag from
  // originalConn

	PV::PVParams* params = parent->parameters();

  sharedWeights = originalConn->usingSharedWeights();
  params->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

  numAxonalArborLists = originalConn->numberOfAxonalArborLists();
  params->handleUnnecessaryParameter(name, "numAxonalArbors",
                                     numAxonalArborLists);

  shrinkPatches_flag = originalConn->getShrinkPatches_flag();
  parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches",
                                                   shrinkPatches_flag);
  return PV_SUCCESS;
}
}
