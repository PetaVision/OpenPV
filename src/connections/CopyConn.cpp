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

CopyConn::CopyConn(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int CopyConn::initialize_base() {
   originalConnName = NULL;
   originalConn = NULL;
   return PV_SUCCESS;
}

int CopyConn::initialize(char const * name, HyPerCol * hc) {
   return HyPerConn::initialize(name, hc);
}

int CopyConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return PV_SUCCESS;
}

void CopyConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   // CopyConn doesn't use a weight initializer
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void CopyConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   // During the communication phase, numAxonalArbors will be copied from originalConn
}

void CopyConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the communication phase, plasticityFlag will be copied from originalConn
}

void CopyConn::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      triggerFlag = false; // make sure that CopyConn always checks if its originalConn has updated
      parent->parameters()->handleUnnecessaryParameter(name, "triggerFlag", triggerFlag);
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

void CopyConn::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalConnName", &originalConnName);
}

int CopyConn::communicateInitInfo() {
   int status = PV_SUCCESS;
   BaseConnection * originalConnBase = parent->getConnFromName(this->originalConnName);
   if (originalConnBase==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalConnName \"%s\" does not refer to any connection in the column.\n", parent->parameters()->groupKeywordFromName(name), name, this->originalConnName);
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
         const char * connectiontype = parent->parameters()->groupKeywordFromName(name);
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
   size_t arborsize = (size_t) (xPatchSize() * yPatchSize() * fPatchSize() * getNumDataPatches());
   memcpy(get_wDataStart(arborId), originalConn->get_wDataStart(arborId), arborsize);
   return PV_SUCCESS;
}

CopyConn::~CopyConn() {
   free(originalConnName);
}

} /* namespace PV */
