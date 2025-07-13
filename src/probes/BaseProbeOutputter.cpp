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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

BaseProbeOutputter::~BaseProbeOutputter() {}

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
   if (!mMessageParam.empty()) {
      mMessageString = mMessageParam + ":";
   }
}

void BaseProbeOutputter::initOutputStreams(Checkpointer *checkpointer, int localNBatch) {
   mLocalNBatch = localNBatch;
   if (!mTextOutputFlag or mProbeOutputFilename.empty()) {
      return;
   }

   if (mIOMPIBlock->getRank() == 0 and mCommunicator->commRank() == 0) {
      int blockBatchDimension = mIOMPIBlock->getBatchDimension();
      int numBatchElements    = mLocalNBatch * blockBatchDimension;
      mOutputStreams.resize(numBatchElements);
      int globalBatchOffset = calcGlobalBatchOffset();
      // The global batch indices managed by this IOBlock are
      // {globalBatchOffset, globalBatchOffset + 1, ..., globalBatchOffset + LocalNBatch - 1}.

      std::string dir  = dirName(mProbeOutputFilename);
      std::string base = stripExtension(mProbeOutputFilename);
      std::string ext  = extension(mProbeOutputFilename);
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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ProbeComponent::initialize(params, defaults);
   mCommunicator = comm;
   mIOMPIBlock   = comm->getIOMPIBlock();
}

void BaseProbeOutputter::ioParam_message(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("textOutputFlag"));
   if (mTextOutputFlag) {
      mParamsIO->ioParam(ioSwitch, "message", &mMessageParam, false /*warnIfAbsent*/);
      if (ioSwitch == ParamsIOSwitch::Read) {
         initMessageString();
      }
   }
}

void BaseProbeOutputter::ioParam_probeOutputFile(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("textOutputFlag"));
   if (mTextOutputFlag) {
      bool warnIfAbsentFlag = false;
      mParamsIO->ioParam(ioSwitch, "probeOutputFile", &mProbeOutputFilename, warnIfAbsentFlag);
   }
}

void BaseProbeOutputter::ioParam_textOutputFlag(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "textOutputFlag", &mTextOutputFlag);
}

void BaseProbeOutputter::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_textOutputFlag(ioSwitch);
   ioParam_probeOutputFile(ioSwitch);
   ioParam_message(ioSwitch);
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
      pvAssert(mProbeOutputFilename.empty());
      return std::make_shared<PrintStream>(PV::getOutputStream());
   }
   else {
      pvAssert(!mProbeOutputFilename.empty());
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
