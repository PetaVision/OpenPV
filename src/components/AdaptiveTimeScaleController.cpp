/*
 * AdaptiveTimeScaleController.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include "AdaptiveTimeScaleController.hpp"
#include "arch/mpi/mpi.h"
#include "include/pv_common.h"
#include "io/FileStream.hpp"
#include "io/fileio.hpp"
#include "utils/PVLog.hpp"

namespace PV {

AdaptiveTimeScaleController::AdaptiveTimeScaleController(
      char const *name,
      int batchWidth,
      double baseMax,
      double baseMin,
      double tauFactor,
      double growthFactor,
      Communicator const *communicator) {
   mName                     = strdup(name);
   mBatchWidth               = batchWidth;
   mBaseMax                  = baseMax;
   mBaseMin                  = baseMin;
   mTauFactor                = tauFactor;
   mGrowthFactor             = growthFactor;
   mCommunicator             = communicator;

   mTimeScaleInfo.resize(batchWidth);
   mOldTimeScaleInfo.resize(batchWidth);
   for (int b = 0; b < batchWidth; ++b) {
      TimeScaleData &a = mTimeScaleInfo[b];
      a.mTimeScale = baseMin;
      a.mTimeScaleMax = baseMax;
      a.mTimeScaleTrue = -1.0;
   }
   for (int b = 0; b < batchWidth; ++b) {
      TimeScaleData &a = mOldTimeScaleInfo[b];
      a.mTimeScale = baseMin;
      a.mTimeScaleTrue = -1.0;
   }
   setDescription(std::string("AdaptiveTimeScaleController \"") + mName + "\"");
}

AdaptiveTimeScaleController::~AdaptiveTimeScaleController() { free(mName); }

Response::Status AdaptiveTimeScaleController::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer = message->mDataRegistry;
   auto ptr           = std::make_shared<CheckpointEntryTimeScaleInfo>(
         mName, "timescaleinfo", &mTimeScaleInfo.front(), mBatchWidth);
   checkpointer->registerCheckpointEntry(ptr, false /*not constant*/);
   return Response::SUCCESS;
}

std::vector<TimeScaleData> const &AdaptiveTimeScaleController::calcTimesteps(std::vector<double> const &timeScales) {
   FatalIf(
         static_cast<int>(timeScales.size()) != mBatchWidth,
         "new timeScaleData has different size than old timeScaleData (%d versus %d)\n",
         static_cast<int>(mTimeScaleInfo.size()),
         mBatchWidth);
   mOldTimeScaleInfo = mTimeScaleInfo;
   for (int b = 0; b < mBatchWidth; ++b) {
      mTimeScaleInfo[b].mTimeScaleTrue = timeScales[b];
      double E_dt = mTimeScaleInfo[b].mTimeScaleTrue;
      double E_0  = mOldTimeScaleInfo[b].mTimeScaleTrue;
      if (E_dt == E_0) { continue; }

      double dE_dt_scaled = (E_0 - E_dt) / mTimeScaleInfo[b].mTimeScale;

      if ((dE_dt_scaled < 0.0) or (E_0 <= 0.0) or (E_dt <= 0.0)) {
         mTimeScaleInfo[b].mTimeScale    = mBaseMin;
         mTimeScaleInfo[b].mTimeScaleMax = mBaseMax;
      }
      else {
         double tau_eff_scaled = E_0 / dE_dt_scaled;
         mTimeScaleInfo[b].mTimeScale = mTauFactor * tau_eff_scaled;
         if (mTimeScaleInfo[b].mTimeScale >= mTimeScaleInfo[b].mTimeScaleMax) {
            mTimeScaleInfo[b].mTimeScale = mTimeScaleInfo[b].mTimeScaleMax;
            mTimeScaleInfo[b].mTimeScaleMax *= (1 + mGrowthFactor);
         }
      }
   }
   return mTimeScaleInfo;
}

void CheckpointEntryTimeScaleInfo::write(
      std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
   int mpiTag = 150; // chosen arbitrarily; there's no significance to this number
   std::shared_ptr<MPIBlock const> mpiBlock = fileManager->getMPIBlock();
   if (fileManager->isRoot()) {
      // mBatchSize is the batch size on one process. We need to get all the batch elements of
      // an IO block onto the root process of that block.
      int numBatchProcesses        = mpiBlock->getBatchDimension();
      std::vector<double> gatheredData(3 * mBatchSize * numBatchProcesses);
      for (std::size_t b = 0; b < mBatchSize; b++) {
         gatheredData[3 * b]     = mTimeScaleDataPtr[b].mTimeScale;
         gatheredData[3 * b + 1] = mTimeScaleDataPtr[b].mTimeScaleTrue;
         gatheredData[3 * b + 2] = mTimeScaleDataPtr[b].mTimeScaleMax;
      }
      for (int m = 1; m < numBatchProcesses; m++) {
         int sourceRank = mpiBlock->calcRankFromRowColBatch(0, 0, m);
         MPI_Recv(
               &gatheredData[3 * mBatchSize * m],
               3 * mBatchSize,
               MPI_DOUBLE,
               sourceRank,
               mpiTag,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
      }

      std::string filename = generateFilename(std::string("bin"));
      auto fileStream  = fileManager->open(filename.c_str(), std::ios_base::out, verifyWritesFlag);
      fileStream->write(gatheredData.data(), sizeof(double) * gatheredData.size());
      filename = generateFilename(std::string("txt"));
      fileStream = fileManager->open(filename.c_str(), std::ios_base::out, verifyWritesFlag);
      int kb0 = mpiBlock->getBatchIndex() * mBatchSize;
      for (std::size_t b = 0; b < mBatchSize * numBatchProcesses; b++) {
         *fileStream << "batch index = " << b + kb0 << "\n";
         *fileStream << "time = " << simTime << "\n";
         *fileStream << "timeScale = " << gatheredData[3 * b] << "\n";
         *fileStream << "timeScaleTrue = " << gatheredData[3 * b + 1] << "\n";
         *fileStream << "timeScaleMax = " << gatheredData[3 * b + 2] << "\n";
      }
   }
   else if (mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0) {
      pvAssert(mpiBlock->getBatchIndex() != 0); // getBatchIndex()==0 case in if-block above
      std::vector<double> dataToSend(3 * mBatchSize);
      for (std::size_t b = 0; b < mBatchSize; b++) {
         dataToSend[3 * b]     = mTimeScaleDataPtr[b].mTimeScale;
         dataToSend[3 * b + 1] = mTimeScaleDataPtr[b].mTimeScaleTrue;
         dataToSend[3 * b + 2] = mTimeScaleDataPtr[b].mTimeScaleMax;
      }
      MPI_Send(
            dataToSend.data(),
            3 * mBatchSize,
            MPI_DOUBLE,
            0,
            mpiTag,
            mpiBlock->getComm());
   }
}

void CheckpointEntryTimeScaleInfo::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   int mpiTag                   = 150; // chosen arbitrarily; there's no significance to this number
   int numBatchProcesses        = fileManager->getMPIBlock()->getBatchDimension();
   std::vector<double> gatheredData(3 * (int)mBatchSize * numBatchProcesses);
   std::shared_ptr<MPIBlock const> mpiBlock = fileManager->getMPIBlock();
   if (mpiBlock->getRank() == 0) {
      std::string filename = generateFilename(std::string("bin"));
      auto fileStream = fileManager->open(filename.c_str(), std::ios_base::in, false);
      fileStream->read(gatheredData.data(), sizeof(double) * gatheredData.size());
   }
   MPI_Bcast(gatheredData.data(), gatheredData.size(), MPI_DOUBLE, 0, mpiBlock->getComm());
   int batchProcessIndex = mpiBlock->getBatchIndex();
   pvAssert(batchProcessIndex >= 0 and batchProcessIndex < numBatchProcesses);
   for (std::size_t b = 0; b < mBatchSize; b++) {
      int blockBatchIndex                  = batchProcessIndex * mBatchSize + b;
      mTimeScaleDataPtr[b].mTimeScale     = gatheredData[3 * blockBatchIndex];
      mTimeScaleDataPtr[b].mTimeScaleTrue = gatheredData[3 * blockBatchIndex + 1];
      mTimeScaleDataPtr[b].mTimeScaleMax  = gatheredData[3 * blockBatchIndex + 2];
   }
}

void CheckpointEntryTimeScaleInfo::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, std::string("bin"));
   deleteFile(fileManager, std::string("txt"));
}

} /* namespace PV */
