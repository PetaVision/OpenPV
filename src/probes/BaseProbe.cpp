/*
 * BaseProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "BaseProbe.hpp"
#include "arch/mpi/mpi.h"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "checkpointing/CheckpointEntryMPIRecvStream.hpp"
#include "io/FileManager.hpp"
#include "io/FileStreamBuilder.hpp"
#include "io/MPISendStream.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/PathComponents.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace PV {

int BaseProbe::mNumProbes = 0; // Initialize number of probes overall, a static BaseProbes member.

BaseProbe::BaseProbe() {
   initialize_base();
   // Derived classes of BaseProbe should call BaseProbe::initialize themselves.
}

BaseProbe::~BaseProbe() {
   flushOutputStreams();
   mOutputStreams.clear();
   std::free(targetName);
   targetName = nullptr;
   std::free(msgparams);
   msgparams = nullptr;
   std::free(msgstring);
   msgstring = nullptr;
   std::free(mProbeOutputFilename);
   mProbeOutputFilename = nullptr;
   if (triggerLayerName) {
      std::free(triggerLayerName);
      triggerLayerName = nullptr;
   }
   mMPIRecvStreams.clear();
   delete mInitialIOTimer;
   delete mInitialIOWaitTimer;
   delete mIOTimer;
   delete mIOWaitTimer;
}

int BaseProbe::initialize_base() {
   targetName       = nullptr;
   msgparams        = nullptr;
   msgstring        = nullptr;
   textOutputFlag   = true;
   triggerFlag      = false;
   triggerLayerName = nullptr;
   triggerOffset    = 0;
   lastUpdateTime   = 0.0;
   mProbeValues.clear();
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
void BaseProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

int BaseProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_targetName(ioFlag);
   ioParam_message(ioFlag);
   ioParam_textOutputFlag(ioFlag);
   ioParam_probeOutputFile(ioFlag);
   ioParam_statsFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   return PV_SUCCESS;
}

void BaseProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void BaseProbe::ioParam_message(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(ioFlag, name, "message", &msgparams, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      initMessage(msgparams);
   }
}

void BaseProbe::ioParam_textOutputFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "textOutputFlag", &textOutputFlag, textOutputFlag);
}

void BaseProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "textOutputFlag"));
   if (textOutputFlag) {
      parameters()->ioParamString(
            ioFlag,
            name,
            "probeOutputFile",
            &mProbeOutputFilename,
            nullptr,
            false /*warnIfAbsent*/);
   }
}

void BaseProbe::ioParam_statsFlag(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "textOutputFlag"));
   if (textOutputFlag) {
      parameters()->ioParamValue(
            ioFlag, name, "statsFlag", &mStatsFlag, mStatsFlag, false /*warnIfAbsent*/);
   }
}

void BaseProbe::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      triggerFlag = (triggerLayerName != NULL && triggerLayerName[0] != '\0');
   }
}

void BaseProbe::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parameters()->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if (triggerOffset < 0) {
         Fatal().printf(
               "%s \"%s\" error in rank %d process: TriggerOffset (%f) "
               "must be positive\n",
               parameters()->groupKeywordFromName(name),
               name,
               mCommunicator->globalCommRank(),
               triggerOffset);
      }
   }
}

int BaseProbe::calcGlobalBatchOffset() {
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   return (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex()) * mLocalBatchWidth;
}

void BaseProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<PrepareCheckpointWriteMessage const>(msgptr);
      return respondPrepareCheckpointWrite(castMessage);
   };
   mMessageActionMap.emplace("PrepareCheckpointWrite", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

void BaseProbe::initOutputStreams(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (mStatsFlag) {
      initOutputStreamsStatsFlag(message);
   }
   else {
      initOutputStreamsByBatchElement(message);
   }
}

void BaseProbe::initOutputStreamsStatsFlag(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer     = message->mDataRegistry;
   auto globalFileManager = std::make_shared<FileManager>(
         getCommunicator()->getIOMPIBlock(),
         getCommunicator()->getOutputFileManager()->getBaseDirectory());
   if (globalFileManager->isRoot()) {
      mOutputStreams.resize(1);
      if (getProbeOutputFilename() and getProbeOutputFilename()[0]) {
         bool createFlag = checkpointer->getCheckpointReadDirectory().empty();
         auto fileStream = FileStreamBuilder(
                                 globalFileManager,
                                 getProbeOutputFilename(),
                                 true /*text*/,
                                 false /*not read-only*/,
                                 createFlag,
                                 checkpointer->doesVerifyWrites())
                                 .get();
         auto checkpointEntry = std::make_shared<CheckpointEntryFilePosition>(
               getProbeOutputFilename(), std::string("filepos"), fileStream);
         bool registerSucceeded = checkpointer->registerCheckpointEntry(
               checkpointEntry, false /*not constant for entire run*/);
         FatalIf(
               !registerSucceeded,
               "%s failed to register %s for checkpointing.\n",
               getDescription_c(),
               checkpointEntry->getName().c_str());
         mOutputStreams[0] = fileStream;
      }
      else {
         mOutputStreams[0] = std::make_shared<PrintStream>(PV::getOutputStream());
      }
   }
   else {
      mOutputStreams.clear();
   }
}

void BaseProbe::initOutputStreamsByBatchElement(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer = message->mDataRegistry;
   if (isBatchBaseProc()) {
      auto ioMPIBlock      = getCommunicator()->getIOMPIBlock();
      int mpiBatchIndex    = ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex();
      int localBatchOffset = mLocalBatchWidth * mpiBatchIndex;
      mOutputStreams.resize(mLocalBatchWidth);
      if (isWritingToFile()) {
         if (isRootProc()) {
            std::string probeOutputFilename(mProbeOutputFilename);
            std::string dir      = dirName(probeOutputFilename);
            std::string base     = stripExtension(probeOutputFilename);
            std::string ext      = extension(probeOutputFilename);
            std::string pathRoot = dir + "/" + base + "_batchElement_";

            int blockBatchSize = ioMPIBlock->getBatchDimension() * mLocalBatchWidth;
            // set up MPIRecvStream objects for batch elements that are not on the root process
            mMPIRecvStreams.reserve(blockBatchSize - mLocalBatchWidth);
            initializeTagVectors(blockBatchSize - mLocalBatchWidth, mTagSpacing);
            auto fileManager = getCommunicator()->getOutputFileManager();

            bool createFlag = checkpointer->getCheckpointReadDirectory().empty();
            for (int b = 0; b < blockBatchSize; ++b) {
               int batchProcessIndex = b / mLocalBatchWidth; // integer division
               int sendingRank       = ioMPIBlock->calcRankFromRowColBatch(0, 0, batchProcessIndex);
               if (sendingRank == ioMPIBlock->getRank()) {
                  int localBatchIndex  = b % mLocalBatchWidth;
                  int globalBatchIndex = localBatchIndex + localBatchOffset;
                  auto path            = pathRoot + std::to_string(globalBatchIndex) + ext;
                  bool verifyWrites    = checkpointer->doesVerifyWrites();
                  auto fileStream      = FileStreamBuilder(
                                          fileManager,
                                          path,
                                          true /*text*/,
                                          false /*not read-only*/,
                                          createFlag,
                                          verifyWrites)
                                          .get();
                  auto checkpointEntry =
                        std::make_shared<CheckpointEntryFilePosition>(path, "filepos", fileStream);
                  bool registerSucceeded = checkpointer->registerCheckpointEntry(
                        checkpointEntry, false /*not constant for entire run*/);
                  FatalIf(
                        !registerSucceeded,
                        "%s failed to register %s for checkpointing.\n",
                        getDescription_c(),
                        checkpointEntry->getName().c_str());
                  mOutputStreams[localBatchIndex] = fileStream;
               }
               else {
                  int globalBatchIndex = b + localBatchOffset;
                  auto batchPath       = pathRoot + std::to_string(globalBatchIndex) + ext;
                  std::string checkpointPath(batchPath + "_filepos");
                  batchPath = fileManager->makeBlockFilename(batchPath);
                  mMPIRecvStreams.emplace_back(
                        batchPath, ioMPIBlock->getComm(), sendingRank, createFlag);
                  auto checkpointEntry = std::make_shared<CheckpointEntryMPIRecvStream>(
                        checkpointPath, mMPIRecvStreams.back());
                  bool constantEntireRunFlag = false;
                  checkpointer->registerCheckpointEntry(checkpointEntry, constantEntireRunFlag);
               }
            }
         }
         else { // ioMPIBlock->getRank() != 0; use MPISendStream
            initializeTagVectors(mLocalBatchWidth, mTagSpacing);
            for (int b = 0; b < mLocalBatchWidth; b++) {
               mOutputStreams[b] =
                     std::make_shared<MPISendStream>(ioMPIBlock->getComm(), 0 /*receiving rank*/);
            }
         }
      }
      else { // no ProbeOutputFilename; use default output stream.
         for (int b = 0; b < mLocalBatchWidth; b++) {
            mOutputStreams[b] = std::make_shared<PrintStream>(PV::getOutputStream());
         }
      }
   }
   else {
      mOutputStreams.clear();
   }
}

Response::Status
BaseProbe::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const>(message)) {
   writeParams();
   return Response::SUCCESS;
}

void BaseProbe::initNumValues() { setNumValues(mLocalBatchWidth); }

void BaseProbe::setNumValues(int n) {
   if (n > 0) {
      mProbeValues.resize(n);
   }
   else {
      mProbeValues.clear();
   }
}

Response::Status
BaseProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Retrieve local batch width and set up number of values.
   int const nBatchGlobal = message->mNBatchGlobal;
   mLocalBatchWidth       = nBatchGlobal / mCommunicator->numCommBatches();
   pvAssert(mLocalBatchWidth * mCommunicator->numCommBatches() == nBatchGlobal);
   initNumValues();
   auto *objectTable = message->mObjectTable;

   // Set up triggering.
   if (triggerFlag) {
      auto triggerLayer = objectTable->findObject<HyPerLayer>(triggerLayerName);
      FatalIf(
            triggerLayer == nullptr,
            "%s triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
            getDescription_c(),
            triggerLayerName);
      mTriggerControl = triggerLayer->getComponentByType<LayerUpdateController>();
      FatalIf(
            mTriggerControl == nullptr,
            "%s triggerLayer \"%s\" does not have a LayerUpdateController component.\n",
            getDescription_c(),
            triggerLayerName);
   }
   return Response::SUCCESS;
}

int BaseProbe::initMessage(const char *msg) {
   assert(msgstring == NULL);
   int status = PV_SUCCESS;
   if (msg != NULL && msg[0] != '\0') {
      size_t msglen   = strlen(msg);
      this->msgstring = (char *)std::calloc(
            msglen + 2,
            sizeof(char)); // Allocate room for colon plus null terminator
      if (this->msgstring) {
         std::memcpy(this->msgstring, msg, msglen);
         this->msgstring[msglen]     = ':';
         this->msgstring[msglen + 1] = '\0';
      }
   }
   else {
      this->msgstring = (char *)std::calloc(1, sizeof(char));
      if (this->msgstring) {
         this->msgstring[0] = '\0';
      }
   }
   if (!this->msgstring) {
      ErrorLog().printf(
            "%s \"%s\": Unable to allocate memory for probe's message.\n",
            parameters()->groupKeywordFromName(name),
            name);
      status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   return status;
}

bool BaseProbe::needUpdate(double simTime, double dt) const {
   if (triggerFlag) {
      return mTriggerControl->needUpdate(simTime + triggerOffset, dt);
   }
   return true;
}

Response::Status
BaseProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   FatalIf(
         getCommunicator()->getIOMPIBlock() == nullptr,
         "\"%s\" called with null I/O MPIBlock\n",
         getDescription().c_str());
   initOutputStreams(message);

   auto *checkpointer = message->mDataRegistry;
   mInitialIOTimer    = new Timer(getName(), "probe", "initialio ");
   checkpointer->registerTimer(mInitialIOTimer);
   mInitialIOWaitTimer = new Timer(getName(), "probe", "initiowait");
   checkpointer->registerTimer(mInitialIOWaitTimer);
   mIOTimer = new Timer(getName(), "probe", "io     ");
   checkpointer->registerTimer(mIOTimer);
   mIOWaitTimer = new Timer(getName(), "probe", "iowait ");
   checkpointer->registerTimer(mIOWaitTimer);

   return Response::SUCCESS;
}

void BaseProbe::getValues(double timevalue) {
   if (needRecalc(timevalue)) {
      calcValues(timevalue);
      lastUpdateTime = referenceUpdateTime(timevalue);
   }
}

void BaseProbe::getValues(double timevalue, double *values) {
   getValues(timevalue);
   std::memcpy(values, mProbeValues.data(), sizeof(*mProbeValues.data()) * mProbeValues.size());
}

void BaseProbe::getValues(double timevalue, std::vector<double> *valuesVector) {
   valuesVector->resize(this->getNumValues());
   getValues(timevalue, &valuesVector->front());
}

Response::Status BaseProbe::outputStateWrapper(double simTime, double dt) {
   auto status = Response::NO_ACTION;
   if (textOutputFlag && needUpdate(simTime, dt)) { // Hopefully don't need a Timer on needUpdate
      Timer *timer = simTime ? mIOTimer : mInitialIOTimer;
      if (mStatsFlag) {
         timer->start();
         status = outputStateStats(simTime, dt);
         timer->stop();
      }
      else {
         timer->start();
         status = outputState(simTime, dt);
         timer->stop();
         Timer *timer = simTime ? mIOWaitTimer : mInitialIOWaitTimer;
         timer->start();
         transferMPIOutput();
         timer->stop();
      }
   }
   return status;
}

void BaseProbe::flushOutputStreams() {
   if (mStatsFlag) {
      return;
   }
   if (!isWritingToFile()) {
      return;
   }
   transferMPIOutput();
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   if (isBatchBaseProc()) {
      if (isRootProc()) {
         std::vector<int> nonRootTags(mCurrentTag.size());
         for (int p = 0; p < ioMPIBlock->getBatchDimension(); ++p) {
            int sendingRank = ioMPIBlock->calcRankFromRowColBatch(0, 0, p);
            if (p == ioMPIBlock->getBatchIndex()) {
               continue;
            }
            int tagVectorOffset = p - (p > ioMPIBlock->getBatchIndex() ? 1 : 0);
            tagVectorOffset *= mLocalBatchWidth;
            pvAssert(tagVectorOffset < static_cast<int>(nonRootTags.size()));
            MPI_Recv(
                  &nonRootTags.at(tagVectorOffset),
                  mLocalBatchWidth,
                  MPI_INT,
                  sendingRank,
                  4999 /*tag*/,
                  ioMPIBlock->getComm(),
                  MPI_STATUS_IGNORE);
         }
         auto vsize = nonRootTags.size();
         pvAssert(vsize == mCurrentTag.size());
         for (std::vector<int>::size_type v = 0; v < vsize; ++v) {
            while (mCurrentTag[v] != nonRootTags[v]) {
               int bytesReceived = mMPIRecvStreams[v].receive(mCurrentTag[v]);
               if (bytesReceived > 0) {
                  int newTag = incrementTag(v);
               }
            }
         }
      }
      else {
         pvAssert(static_cast<int>(mCurrentTag.size() == mLocalBatchWidth));
         MPI_Send(
               mCurrentTag.data(),
               mLocalBatchWidth,
               MPI_INT,
               0 /*destination rank*/,
               4999 /*tag*/,
               ioMPIBlock->getComm());
      }
   }
}

int BaseProbe::incrementTag(int index) {
   int tag            = mCurrentTag.at(index);
   int newTag         = tag + 1;
   newTag             = (newTag >= mTagLimit[index]) ? mStartTag[index] : newTag;
   mCurrentTag[index] = newTag;
   return newTag;
}

void BaseProbe::initializeTagVectors(int vectorSize, int spacing) {
   mProbeIndex = mNumProbes++;
   mCurrentTag.resize(vectorSize);
   mStartTag.resize(vectorSize);
   mTagLimit.resize(vectorSize);
   for (int b = 0; b < vectorSize; ++b) {
      int localBatchIndex = b % mLocalBatchWidth;
      int startTag        = mBaseTag + spacing * (localBatchIndex + mLocalBatchWidth * mProbeIndex);
      mCurrentTag[b]      = startTag;
      mStartTag[b]        = startTag;
      mTagLimit[b]        = startTag + spacing;
   }
}

bool BaseProbe::isBatchBaseProc() const {
   int blockColumnIndex = mCommunicator->commColumn();
   int blockRowIndex    = mCommunicator->commRow();
   return (blockColumnIndex == 0 and blockRowIndex == 0);
}

bool BaseProbe::isRootProc() const { return getCommunicator()->getIOMPIBlock()->getRank() == 0; }

void BaseProbe::receive(int batchProcessIndex, int localBatchIndex) {
   pvAssert(isRootProc());
   int blockBatchIndex = batchProcessIndex * mLocalBatchWidth + localBatchIndex;
   int streamIndex     = blockBatchIndex;
   auto ioMPIBlock     = getCommunicator()->getIOMPIBlock();
   streamIndex -= (batchProcessIndex > ioMPIBlock->getBatchIndex() ? mLocalBatchWidth : 0);
   int tag           = mCurrentTag[streamIndex];
   auto &recvStream  = mMPIRecvStreams[streamIndex];
   int bytesReceived = recvStream.receive(tag);
   if (bytesReceived > 0) {
      incrementTag(streamIndex);
   }
}

void BaseProbe::transferMPIOutput() {
   if (mStatsFlag) {
      return;
   }
   if (!isWritingToFile()) {
      return;
   }
   if (isBatchBaseProc()) {
      if (isRootProc()) {
         auto ioMPIBlock    = getCommunicator()->getIOMPIBlock();
         int blockBatchSize = ioMPIBlock->getBatchDimension() * mLocalBatchWidth;
         for (int b = 0; b < blockBatchSize; ++b) {
            int batchProcessIndex = b / mLocalBatchWidth; // integer division
            if (batchProcessIndex == ioMPIBlock->getBatchIndex()) {
               continue;
            }
            int localBatchIndex = b % mLocalBatchWidth;
            receive(batchProcessIndex, localBatchIndex);
         }
      }
      else {
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            auto *stream = dynamic_cast<MPISendStream *>(&output(b));
            pvAssert(stream != nullptr);
            int tag       = mCurrentTag[b];
            int bytesSent = stream->send(tag);
            if (bytesSent > 0) {
               incrementTag(b);
            }
         }
      }
   }
}

Response::Status BaseProbe::prepareCheckpointWrite(double simTime) {
   if (!mStatsFlag) {
      flushOutputStreams();
   }
   return Response::SUCCESS;
}

} // namespace PV
