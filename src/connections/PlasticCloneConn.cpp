/* PlasticCloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "PlasticCloneConn.hpp"

namespace PV {

PlasticCloneConn::PlasticCloneConn() { initialize_base(); }

PlasticCloneConn::PlasticCloneConn(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int PlasticCloneConn::initialize_base() { return PV_SUCCESS; }

int PlasticCloneConn::initialize(const char *name, HyPerCol *hc) {
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

void PlasticCloneConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
   // During the communication phase, weightUpdatePeriod will be copied
   // from originalConn
}

void PlasticCloneConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime");
   }
   // During the communication phase, initialWeightUpdateTime will be copied
   // from originalConn
}

void PlasticCloneConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax");
   }
   // During the communication phase, dWMax will be copied from
   // originalConn
}

void PlasticCloneConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized");
   }
   // During the communication phase, keepKernelsSynchronized will be copied
   // from originalConn
}

void PlasticCloneConn::ioParam_normalizeDw(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "normalizeDw");
   }
   // During the communication phase, normalizeDw will be copied from
   // originalConn
}

int PlasticCloneConn::communicateInitInfo() {
   int status = CloneConn::communicateInitInfo();
   originalConn->addClone(this);
   normalizeDwFlag = originalConn->getNormalizeDwFlag();

   return status;
}

int PlasticCloneConn::constructWeights() {
   int status = CloneConn::constructWeights();
   if (status == PV_SUCCESS) {
      // Additionally point activations to orig conn
      numKernelActivations = this->originalConn->get_activations();
   }
   return status;
}

int PlasticCloneConn::deleteWeights() {
   wPatches             = NULL;
   wDataStart           = NULL;
   gSynPatchStart       = NULL;
   aPostOffset          = NULL;
   dwDataStart          = NULL;
   numKernelActivations = NULL;
   return 0;
}

int PlasticCloneConn::cloneParameters() {
   // called by CloneConn::communicateInitInfo, before it calls
   // HyPerConn::communicateInitInfo
   CloneConn::cloneParameters();

   // CloneConn set plasticity flag to false; PlasticCloneConn needs it to be true.
   plasticityFlag               = true;
   weightUpdatePeriod           = originalConn->getWeightUpdatePeriod();
   initialWeightUpdateTime      = originalConn->getWeightUpdatePeriod();
   dWMax                        = originalConn->getDWMax();
   keepKernelsSynchronized_flag = originalConn->getKeepKernelsSynchronized();
   normalizeDwFlag              = originalConn->getNormalizeDwFlag();

   return PV_SUCCESS;
}

PlasticCloneConn::~PlasticCloneConn() { deleteWeights(); }

} // end namespace PV
