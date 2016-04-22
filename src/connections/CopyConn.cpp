/*
 * CopyConn.cpp
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#include "CopyConn.hpp"

namespace PV {

CopyConn::CopyConn() {
   initialize_base();
}

CopyConn::CopyConn(char const * name, HyPerCol * hc, NormalizeBase * weightNormalizer) {
   initialize_base();
   initialize(name, hc, weightNormalizer);
}

int CopyConn::initialize_base() {
   originalConnName = NULL;
   originalConn = NULL;
   return PV_SUCCESS;
}

int CopyConn::initialize(char const * name, HyPerCol * hc, NormalizeBase * weightNormalizer) {
   return HyPerConn::initialize(name, hc, NULL, weightNormalizer);
}

int CopyConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return PV_SUCCESS;
}

void CopyConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   // CopyConn determines sharedWeights from originalConn, during communicateInitInfo
}

void CopyConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized");
   }
   // CopyConn doesn't use the keepKernelsSynchronized flag.
   // It copies weights from original conn, so kernels will automatically synchronize
   // whenever original the conn synchronizes.
}

void CopyConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // CopyConn doesn't use a weight initializer
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void CopyConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   // CopyConn determines nxp from originalConn, during communicateInitInfo
}

void CopyConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   // CopyConn determines nyp from originalConn, during communicateInitInfo
}

void CopyConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   // CopyConn determines nfp from originalConn, during communicateInitInfo
}

void CopyConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
   // During the setInitialValues phase, the conn will be copied from the original conn, so initializeFromCheckpointFlag is not needed.
}

void CopyConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void CopyConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void CopyConn::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      // make sure that TransposePoolingConn always checks if its originalConn has updated
      triggerFlag = false;
      triggerLayerName = NULL;
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
      parent->parameters()->handleUnnecessaryStringParameter(name, "triggerLayerName", NULL);
   }
}

void CopyConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = parent->getDeltaTime();
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod");
   }
}

void CopyConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = parent->getStartTime();
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime");
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void CopyConn::ioParam_dWMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      // since CopyConn doesn't do its own learning, it doesn't use dWMax
      parent->parameters()->handleUnnecessaryParameter(name, "dWMax");
   }
}

void CopyConn::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      useMask = false; // since CopyConn doesn't do its own learning, it doesn't need to have a mask
      parent->parameters()->handleUnnecessaryParameter(name, "useMask", useMask);
   }
}

void CopyConn::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      maskLayerName = NULL; // since CopyConn doesn't do its own learning, it doesn't need to have a mask
      parent->parameters()->handleUnnecessaryStringParameter(name, "maskLayerName", maskLayerName);
   }
}

void CopyConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int CopyConn::communicateInitInfo() {
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
         fprintf(stderr, "CopyConn \"%s\" error: originalConnName \"%s\" is not an existing connection.\n", name, originalConnName);
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

   plasticityFlag = originalConn->getPlasticityFlag();
   parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);

   status = HyPerConn::communicateInitInfo();

   return status;
}

int CopyConn::setPatchSize() {
   nxp = originalConn->xPatchSize();
   nyp = originalConn->yPatchSize();
   nfp = originalConn->fPatchSize();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", nxp);
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", nyp);
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp);
   return PV_SUCCESS;
}

int CopyConn::setInitialValues() {
   int status = PV_SUCCESS;
   if (originalConn->getInitialValuesSetFlag()) {
      status = HyPerConn::setInitialValues(); // calls initializeWeights
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

PVPatch*** CopyConn::initializeWeights(PVPatch*** patches, pvwdata_t** dataStart) {
   assert(originalConn->getInitialValuesSetFlag()); // setInitialValues shouldn't call this function unless original conn has set its own initial values
   assert(dataStart == get_wDataStart());
   assert(patches==NULL || patches==get_wPatches());
   for (int arbor=0; arbor<numAxonalArborLists; arbor++) {
      copy(arbor);
   }
   return patches;
}

bool CopyConn::needUpdate(double time, double dt) {
   return plasticityFlag && originalConn->getLastUpdateTime() > lastUpdateTime;
}

int CopyConn::updateWeights(int axonID) {
   assert(originalConn->getLastUpdateTime() > lastUpdateTime);
   int status;
   float original_update_time = originalConn->getLastUpdateTime();
   if(original_update_time > lastUpdateTime ) {
      status = copy(axonID);
      lastUpdateTime = parent->simulationTime();
   }
   else
      status = PV_SUCCESS;
   return status;
}  // end of CopyConn::updateWeights(int);

int CopyConn::copy(int arborId) {
   size_t arborsize = (size_t) (xPatchSize() * yPatchSize() * fPatchSize() * getNumDataPatches()) * sizeof(pvwdata_t);
   memcpy(get_wDataStart(arborId), originalConn->get_wDataStart(arborId), arborsize);
   return PV_SUCCESS;
}

CopyConn::~CopyConn() {
   free(originalConnName);
}

BaseObject * createCopyConn(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new CopyConn(name, hc, weightNormalizer);
}

} /* namespace PV */
