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
      bool writeTimeScaleFieldnames,
      Communicator const *communicator) {
   mName                     = strdup(name);
   mBatchWidth               = batchWidth;
   mBaseMax                  = baseMax;
   mBaseMin                  = baseMin;
   mTauFactor                = tauFactor;
   mGrowthFactor             = growthFactor;
   mWriteTimeScaleFieldnames = writeTimeScaleFieldnames;
   mCommunicator             = communicator;

   mTimeScaleInfo.mTimeScale.assign(mBatchWidth, mBaseMin);
   mTimeScaleInfo.mTimeScaleMax.assign(mBatchWidth, mBaseMax);
   mTimeScaleInfo.mTimeScaleTrue.assign(mBatchWidth, -1.0);
   mOldTimeScale.assign(mBatchWidth, mBaseMin);
   mOldTimeScaleTrue.assign(mBatchWidth, -1.0);
   setDescription(std::string("AdaptiveTimeScaleController \"") + mName + "\"");
}

AdaptiveTimeScaleController::~AdaptiveTimeScaleController() { free(mName); }

Response::Status AdaptiveTimeScaleController::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer = message->mDataRegistry;
   auto ptr           = std::make_shared<CheckpointEntryTimeScaleInfo>(
         mName, "timescaleinfo", &mTimeScaleInfo);
   checkpointer->registerCheckpointEntry(ptr, false /*not constant*/);
   return Response::SUCCESS;
}

std::vector<double> AdaptiveTimeScaleController::calcTimesteps(
      double timeValue,
      std::vector<double> const &rawTimeScales) {
   mOldTimeScaleInfo             = mTimeScaleInfo;
   mTimeScaleInfo.mTimeScaleTrue = rawTimeScales;
   for (int b = 0; b < mBatchWidth; b++) {
      double E_dt         = mTimeScaleInfo.mTimeScaleTrue[b];
      double E_0          = mOldTimeScaleInfo.mTimeScaleTrue[b];
      double dE_dt_scaled = (E_0 - E_dt) / mTimeScaleInfo.mTimeScale[b];

      if (E_dt == E_0) {
         continue;
      }

      if ((dE_dt_scaled < 0.0) || (E_0 <= 0) || (E_dt <= 0)) {
         mTimeScaleInfo.mTimeScale[b]    = mBaseMin;
         mTimeScaleInfo.mTimeScaleMax[b] = mBaseMax;
      }
      else {
         double tau_eff_scaled = E_0 / dE_dt_scaled;

         // dt := mTimeScaleMaxBase * tau_eff
         mTimeScaleInfo.mTimeScale[b] = mTauFactor * tau_eff_scaled;
         if (mTimeScaleInfo.mTimeScale[b] >= mTimeScaleInfo.mTimeScaleMax[b]) {
            mTimeScaleInfo.mTimeScale[b]    = mTimeScaleInfo.mTimeScaleMax[b];
            mTimeScaleInfo.mTimeScaleMax[b] = (1 + mGrowthFactor) * mTimeScaleInfo.mTimeScaleMax[b];
         }
      }
   }
   return mTimeScaleInfo.mTimeScale;
}

void AdaptiveTimeScaleController::writeTimestepInfo(
      double timeValue,
      std::vector<PrintStream *> &streams) {
   for (int b = 0; b < mBatchWidth; b++) {
      auto stream = *streams.at(b);
      if (mWriteTimeScaleFieldnames) {
         stream.printf("sim_time = %f\n", timeValue);
         stream.printf(
               "\tbatch = %d, timeScale = %10.8f, timeScaleTrue = %10.8f",
               b,
               mTimeScaleInfo.mTimeScale[b],
               mTimeScaleInfo.mTimeScaleTrue[b]);
      }
      else {
         stream.printf("%f, ", timeValue);
         stream.printf(
               "%d, %10.8f, %10.8f",
               b,
               mTimeScaleInfo.mTimeScale[b],
               mTimeScaleInfo.mTimeScaleTrue[b]);
      }
      if (mWriteTimeScaleFieldnames) {
         stream.printf(", timeScaleMax = %10.8f\n", mTimeScaleInfo.mTimeScaleMax[b]);
      }
      else {
         stream.printf(", %10.8f\n", mTimeScaleInfo.mTimeScaleMax[b]);
      }
      stream.flush();
   }
}

void CheckpointEntryTimeScaleInfo::write(
      std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
   int mpiTag = 150; // chosen arbitrarily; there's no significance to this number
   std::shared_ptr<MPIBlock const> mpiBlock = fileManager->getMPIBlock();
   if (fileManager->isRoot()) {
      std::size_t processBatchSize = mTimeScaleInfoPtr->mTimeScale.size();
      int numBatchProcesses        = mpiBlock->getBatchDimension();
      std::vector<double> gatheredData(3 * processBatchSize * numBatchProcesses);
      for (std::size_t b = 0; b < processBatchSize; b++) {
         gatheredData[3 * b]     = mTimeScaleInfoPtr->mTimeScale.at(b);
         gatheredData[3 * b + 1] = mTimeScaleInfoPtr->mTimeScaleTrue.at(b);
         gatheredData[3 * b + 2] = mTimeScaleInfoPtr->mTimeScaleMax.at(b);
      }
      for (int m = 1; m < numBatchProcesses; m++) {
         int sourceRank = mpiBlock->calcRankFromRowColBatch(0, 0, m);
         MPI_Recv(
               &gatheredData[3 * processBatchSize * m],
               3 * processBatchSize,
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
      int kb0 = mpiBlock->getBatchIndex() * processBatchSize;
      for (std::size_t b = 0; b < processBatchSize * numBatchProcesses; b++) {
         *fileStream << "batch index = " << b + kb0 << "\n";
         *fileStream << "time = " << simTime << "\n";
         *fileStream << "timeScale = " << gatheredData[3 * b] << "\n";
         *fileStream << "timeScaleTrue = " << gatheredData[3 * b + 1] << "\n";
         *fileStream << "timeScaleMax = " << gatheredData[3 * b + 2] << "\n";
      }
   }
   else if (mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0) {
      pvAssert(mpiBlock->getBatchIndex() != 0); // getBatchIndex()==0 case in if-block above
      std::size_t processBatchSize = mTimeScaleInfoPtr->mTimeScale.size();
      std::vector<double> dataToSend(3 * processBatchSize);
      for (std::size_t b = 0; b < processBatchSize; b++) {
         dataToSend[3 * b]     = mTimeScaleInfoPtr->mTimeScale.at(b);
         dataToSend[3 * b + 1] = mTimeScaleInfoPtr->mTimeScaleTrue.at(b);
         dataToSend[3 * b + 2] = mTimeScaleInfoPtr->mTimeScaleMax.at(b);
      }
      MPI_Send(
            dataToSend.data(),
            3 * processBatchSize,
            MPI_DOUBLE,
            0,
            mpiTag,
            mpiBlock->getComm());
   }
}

void CheckpointEntryTimeScaleInfo::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   int mpiTag                   = 150; // chosen arbitrarily; there's no significance to this number
   std::size_t processBatchSize = (int)mTimeScaleInfoPtr->mTimeScale.size();
   int numBatchProcesses        = fileManager->getMPIBlock()->getBatchDimension();
   std::vector<double> gatheredData(3 * (int)processBatchSize * numBatchProcesses);
   std::shared_ptr<MPIBlock const> mpiBlock = fileManager->getMPIBlock();
   if (mpiBlock->getRank() == 0) {
      std::string filename = generateFilename(std::string("bin"));
      auto fileStream = fileManager->open(filename.c_str(), std::ios_base::in, false);
      fileStream->read(gatheredData.data(), sizeof(double) * gatheredData.size());
   }
   MPI_Bcast(gatheredData.data(), gatheredData.size(), MPI_DOUBLE, 0, mpiBlock->getComm());
   int batchProcessIndex = mpiBlock->getBatchIndex();
   pvAssert(batchProcessIndex >= 0 and batchProcessIndex < numBatchProcesses);
   for (std::size_t b = 0; b < processBatchSize; b++) {
      int blockBatchIndex                     = batchProcessIndex * processBatchSize + b;
      mTimeScaleInfoPtr->mTimeScale.at(b)     = gatheredData[3 * blockBatchIndex];
      mTimeScaleInfoPtr->mTimeScaleTrue.at(b) = gatheredData[3 * blockBatchIndex + 1];
      mTimeScaleInfoPtr->mTimeScaleMax.at(b)  = gatheredData[3 * blockBatchIndex + 2];
   }
}

void CheckpointEntryTimeScaleInfo::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, std::string("bin"));
   deleteFile(fileManager, std::string("txt"));
}

} /* namespace PV */
