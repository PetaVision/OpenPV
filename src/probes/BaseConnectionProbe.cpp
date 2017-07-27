/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() { initialize_base(); }

BaseConnectionProbe::BaseConnectionProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

BaseConnectionProbe::~BaseConnectionProbe() {}

int BaseConnectionProbe::initialize_base() {
   targetConn = NULL;
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char *name, HyPerCol *hc) {
   int status = BaseProbe::initialize(name, hc);
   return status;
}

void BaseConnectionProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(ioFlag, name, "targetConnection", &targetName, NULL, false);
   if (targetName == NULL) {
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

int BaseConnectionProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseProbe::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   targetConn = message->lookup<BaseConnection>(std::string(targetName));
   if (targetConn == nullptr) {
      ErrorLog().printf(
            "%s, rank %d process: targetConnection \"%s\" is "
            "not a connection in the column.\n",
            getDescription_c(),
            parent->columnId(),
            targetName);
      status = PV_FAILURE;
   }
   MPI_Barrier(parent->getCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   targetConn->insertProbe(this);
   return status;
}

void BaseConnectionProbe::initOutputStreams(const char *filename, Checkpointer *checkpointer) {
   MPIBlock const *mpiBlock = checkpointer->getMPIBlock();
   if (getMPIBlock()->getRank() == 0) {
      char const *probeOutputFilename = getProbeOutputFilename();
      if (probeOutputFilename) {
         std::string path(probeOutputFilename);
         std::ios_base::openmode mode = std::ios_base::out;
         if (!checkpointer->getCheckpointReadDirectory().empty()) {
            mode |= std::ios_base::app;
         }
         if (path[0] != '/') {
            path = checkpointer->makeOutputPathFilename(path);
         }
         auto stream = new FileStream(path.c_str(), mode, checkpointer->doesVerifyWrites());
         mOutputStreams.push_back(stream);
      }
      else {
         auto stream = new PrintStream(PV::getOutputStream());
         mOutputStreams.push_back(stream);
      }
   }
   else {
      mOutputStreams.clear();
   }
}

} // end of namespace PV
