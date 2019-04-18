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
      Communicator *communicator) {
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
}

AdaptiveTimeScaleController::~AdaptiveTimeScaleController() { free(mName); }

Response::Status AdaptiveTimeScaleController::registerData(Checkpointer *checkpointer) {
   auto ptr = std::make_shared<CheckpointEntryTimeScaleInfo>(
         mName, "timescaleinfo", checkpointer->getMPIBlock(), &mTimeScaleInfo);
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
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   if (getMPIBlock()->getRank() == 0) {
      int batchWidth   = (int)mTimeScaleInfoPtr->mTimeScale.size();
      std::string path = generatePath(checkpointDirectory, "bin");
      FileStream fileStream{path.c_str(), std::ios_base::out, verifyWritesFlag};
      for (int b = 0; b < batchWidth; b++) {
         fileStream.write(&mTimeScaleInfoPtr->mTimeScale.at(b), sizeof(double));
         fileStream.write(&mTimeScaleInfoPtr->mTimeScaleTrue.at(b), sizeof(double));
         fileStream.write(&mTimeScaleInfoPtr->mTimeScaleMax.at(b), sizeof(double));
      }
      path = generatePath(checkpointDirectory, "txt");
      FileStream txtFileStream{path.c_str(), std::ios_base::out, verifyWritesFlag};
      int kb0 = getMPIBlock()->getBatchIndex() * batchWidth;
      for (std::size_t b = 0; b < batchWidth; b++) {
         txtFileStream << "batch index = " << b + kb0 << "\n";
         txtFileStream << "time = " << simTime << "\n";
         txtFileStream << "timeScale = " << mTimeScaleInfoPtr->mTimeScale[b] << "\n";
         txtFileStream << "timeScaleTrue = " << mTimeScaleInfoPtr->mTimeScaleTrue[b] << "\n";
         txtFileStream << "timeScaleMax = " << mTimeScaleInfoPtr->mTimeScaleMax[b] << "\n";
      }
   }
}

void CheckpointEntryTimeScaleInfo::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "bin");
   deleteFile(checkpointDirectory, "txt");
}

void CheckpointEntryTimeScaleInfo::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   int batchWidth = (int)mTimeScaleInfoPtr->mTimeScale.size();
   if (getMPIBlock()->getRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "bin");
      FileStream fileStream{path.c_str(), std::ios_base::in, false};
      for (std::size_t b = 0; b < batchWidth; b++) {
         fileStream.read(&mTimeScaleInfoPtr->mTimeScale.at(b), sizeof(double));
         fileStream.read(&mTimeScaleInfoPtr->mTimeScaleTrue.at(b), sizeof(double));
         fileStream.read(&mTimeScaleInfoPtr->mTimeScaleMax.at(b), sizeof(double));
      }
   }
   MPI_Bcast(
         mTimeScaleInfoPtr->mTimeScale.data(), batchWidth, MPI_DOUBLE, 0, getMPIBlock()->getComm());
   MPI_Bcast(
         mTimeScaleInfoPtr->mTimeScaleTrue.data(),
         batchWidth,
         MPI_DOUBLE,
         0,
         getMPIBlock()->getComm());
   MPI_Bcast(
         mTimeScaleInfoPtr->mTimeScaleMax.data(),
         batchWidth,
         MPI_DOUBLE,
         0,
         getMPIBlock()->getComm());
}

} /* namespace PV */
