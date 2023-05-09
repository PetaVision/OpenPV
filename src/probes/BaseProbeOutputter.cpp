#include "BaseProbeOutputter.hpp"

#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileStreamBuilder.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/PathComponents.hpp"
#include <cstdlib>
#include <memory>
#include <string>

namespace PV {

BaseProbeOutputter::BaseProbeOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   initialize(objName, params, comm);
}

BaseProbeOutputter::~BaseProbeOutputter() {
   free(mProbeOutputFilename);
   free(mMessageParam);
}

int BaseProbeOutputter::calcGlobalBatchOffset() const {
   return (mIOMPIBlock->getStartBatch() + mIOMPIBlock->getBatchIndex()) * mLocalNBatch;
}

void BaseProbeOutputter::flushOutputStreams() {
   for (auto &s : mOutputStreams) {
      s->flush();
   }

}

void BaseProbeOutputter::initMessageString() {
   pvAssert(mMessageString.empty());
   if (mMessageParam != nullptr and mMessageParam[0] != '\0') {
      mMessageString = mMessageParam;
      mMessageString += ":";
   }
}

void BaseProbeOutputter::initOutputStreams(Checkpointer *checkpointer, int localNBatch) {
   mLocalNBatch = localNBatch;
   if (!mTextOutputFlag or !mProbeOutputFilename or !mProbeOutputFilename[0]) {
      return;
   }

   if (mIOMPIBlock->getRank() == 0 and mCommunicator->commRank() == 0) {
      int blockBatchDimension = mIOMPIBlock->getBatchDimension();
      int numBatchElements    = mLocalNBatch * blockBatchDimension;
      mOutputStreams.resize(numBatchElements);
      int globalBatchOffset = calcGlobalBatchOffset();
      // The global batch indices managed by this IOBlock are
      // {globalBatchOffset, globalBatchOffset + 1, ..., globalBatchOffset + LocalNBatch - 1}.

      std::string probeOutputFilename(mProbeOutputFilename);
      std::string dir  = dirName(probeOutputFilename);
      std::string base = stripExtension(probeOutputFilename);
      std::string ext  = extension(probeOutputFilename);
      std::string pathRoot;
      if (dir != ".") {
         pathRoot = dir + "/";
      }
      pathRoot.append(base).append("_batchElement_");

      for (int b = 0; b < numBatchElements; ++b) {
         int globalBatchIndex = globalBatchOffset + b;
         auto fileManager     = mCommunicator->getOutputFileManager();
         auto path            = pathRoot + std::to_string(globalBatchIndex) + ext;
         bool isText          = true;
         bool notReadOnly     = false;
         bool createFlag      = checkpointer->getCheckpointReadDirectory().empty();
         bool verifyWrites    = checkpointer->doesVerifyWrites();
         auto fileStream =
               FileStreamBuilder(fileManager, path, isText, notReadOnly, createFlag, verifyWrites);
         auto checkpointEntry =
               std::make_shared<CheckpointEntryFilePosition>(path, "filepos", fileStream.get());
         bool registerSucceeded = checkpointer->registerCheckpointEntry(
               checkpointEntry, false /*not constant for entire run*/);
         FatalIf(
               !registerSucceeded,
               "Probe \"%s\" failed to register %s for checkpointing.\n",
               getName_c(),
               checkpointEntry->getName().c_str());
         mOutputStreams[b] = fileStream.get();
      }
      if (checkpointer->getCheckpointReadDirectory().empty()) {
         printHeader();
      }
   }
}

void BaseProbeOutputter::initialize(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   ProbeComponent::initialize(objName, params);
   mCommunicator = comm;
   mIOMPIBlock   = comm->getIOMPIBlock();
}

void BaseProbeOutputter::ioParam_message(enum ParamsIOFlag ioFlag) {
   pvAssert(!getParams()->presentAndNotBeenRead(getName_c(), "textOutputFlag"));
   if (mTextOutputFlag) {
      getParams()->ioParamString(
            ioFlag, getName_c(), "message", &mMessageParam, mMessageParam, false /*warnIfAbsent*/);
      if (ioFlag == PARAMS_IO_READ) {
         initMessageString();
      }
   }
}

void BaseProbeOutputter::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   pvAssert(!getParams()->presentAndNotBeenRead(getName_c(), "textOutputFlag"));
   if (mTextOutputFlag) {
      getParams()->ioParamString(
            ioFlag,
            getName_c(),
            "probeOutputFile",
            &mProbeOutputFilename,
            nullptr,
            false /*warnIfAbsent*/);
   }
}

void BaseProbeOutputter::ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamValue(
         ioFlag, getName_c(), "textOutputFlag", &mTextOutputFlag, mTextOutputFlag);
}

void BaseProbeOutputter::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_textOutputFlag(ioFlag);
   ioParam_probeOutputFile(ioFlag);
   ioParam_message(ioFlag);
}

void BaseProbeOutputter::printStringToAll(char const *str) {
   for (auto &s : mOutputStreams) {
      *s << str << "\n";
   }
}

std::shared_ptr<PrintStream> BaseProbeOutputter::returnOutputStream(int b) {
   if (getCommunicator()->commRank() != 0) {
      return nullptr;
   }
   if (mOutputStreams.empty()) {
      pvAssert(!mProbeOutputFilename or mProbeOutputFilename[0] == '\0');
      return std::make_shared<PrintStream>(PV::getOutputStream());
   }
   else {
      pvAssert(mProbeOutputFilename and mProbeOutputFilename[0] != '\0');
      return mOutputStreams.at(b);
   }
}

void BaseProbeOutputter::printStringToAll(std::string const &str) { printStringToAll(str.c_str()); }

void BaseProbeOutputter::writeString(std::string const &str, int batchIndex) {
   auto outputStream = returnOutputStream(batchIndex);
   if (outputStream) {
      outputStream->printf(getMessage().c_str());
      outputStream->printf(str.c_str());
      outputStream->printf("\n");
   }
}

} // namespace PV
