/*
 * AdaptiveTimestepController.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include "io/AdaptiveTimestepController.hpp"
#include "include/pv_common.h"
#include "io/fileio.hpp"
#include "utils/PVLog.hpp"
#include "arch/mpi/mpi.h"

namespace PV {

AdaptiveTimestepController::AdaptiveTimestepController(
      char const * name,
      int batchWidth,
      double deltaTimeBase,
      double timeScaleMaxBase,
      double timeScaleMin,
      double changeTimeScaleMax,
      double changeTimeScaleMin,
      bool writeTimescales,
      bool writeTimeScaleFieldnames,
      Communicator * communicator,
      bool verifyWrites) {
   mName = strdup(name);
   mBatchWidth = batchWidth;
   mDeltaTimeBase = deltaTimeBase;
   mTimeScaleMaxBase = timeScaleMaxBase;
   mTimeScaleMin = timeScaleMin;
   mChangeTimeScaleMax = changeTimeScaleMax;
   mChangeTimeScaleMin = changeTimeScaleMin;
   mWriteTimescales = writeTimescales;
   mWriteTimeScaleFieldnames = writeTimeScaleFieldnames;
   mCommunicator = communicator;
   mVerifyWrites = verifyWrites;

   mTimeScale.assign(mBatchWidth, mTimeScaleMin);
   mTimeScaleMax.assign(mBatchWidth, mTimeScaleMaxBase);
   mTimeScaleTrue.assign(mBatchWidth, -1.0);
   mOldTimeScale.assign(mBatchWidth, mTimeScaleMin);
   mOldTimeScaleTrue.assign(mBatchWidth, -1.0);
}

AdaptiveTimestepController::~AdaptiveTimestepController() {
   free(mName);
}

int AdaptiveTimestepController::checkpointRead(const char * cpDir, double * timeptr) {

   struct timescalemax_struct {
      double mTimeScale; // mTimeScale factor for increasing/decreasing dt
      double mTimeScaleTrue; // true mTimeScale as returned by HyPerLayer::getTimeScaleTrue() typically computed by an adaptTimeScaleController (ColProbe)
      double mTimeScaleMax; //  current maximum allowed value of mTimeScale as returned by HyPerLayer::getTimeScaleMaxPtr()
   };
   struct timescalemax_struct timescalemax[mBatchWidth];

   for(int b = 0; b < mBatchWidth; b++){
      timescalemax[b].mTimeScale = 1;
      timescalemax[b].mTimeScaleTrue = 1;
      timescalemax[b].mTimeScaleMax = 1;
   }
   size_t timescalemax_size = sizeof(struct timescalemax_struct);
   assert(sizeof(struct timescalemax_struct) == sizeof(double) + sizeof(double) + sizeof(double));
   // read mTimeScale info
   if(mCommunicator->commRank()==0 ) {
      char timescalepath[PV_PATH_MAX];
      int chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/%s_timescaleinfo.bin", cpDir, mName);
      if (chars_needed >= PV_PATH_MAX) {
         pvError().printf("HyPerCol::checkpointRead error: path \"%s/timescaleinfo.bin\" is too long.\n", cpDir);
      }
      PV_Stream * timescalefile = PV_fopen(timescalepath,"r",false/*mVerifyWrites*/);
      if (timescalefile == nullptr) {
         pvWarn(errorMessage);
         errorMessage.printf("HyPerCol::checkpointRead: unable to open \"%s\" for reading: %s.\n", timescalepath, strerror(errno));
         errorMessage.printf("    will use default value of mTimeScale=%f, mTimeScaleTrue=%f, mTimeScaleMax=%f, mTimeScaleMax2=%f\n", 1.0, 1.0, 1.0, 1.0);
      }
      else {
         for(int b = 0; b < mBatchWidth; b++){
            long int startpos = getPV_StreamFilepos(timescalefile);
            PV_fread(&timescalemax[b],1,timescalemax_size,timescalefile);
            long int endpos = getPV_StreamFilepos(timescalefile);
            assert(endpos-startpos==(int)sizeof(struct timescalemax_struct));
         }
         PV_fclose(timescalefile);
      }
   }
   //Grab only the necessary part based on comm batch id

   MPI_Bcast(&timescalemax,(int) timescalemax_size*mBatchWidth,MPI_CHAR,0,mCommunicator->communicator());
   for (int b = 0; b < mBatchWidth; b++){
      mTimeScale[b] = timescalemax[b].mTimeScale;
      mTimeScaleTrue[b] = timescalemax[b].mTimeScaleTrue;
      mTimeScaleMax[b] = timescalemax[b].mTimeScaleMax;
   }
   return PV_SUCCESS;
}

int AdaptiveTimestepController::checkpointWrite(const char * cpDir) {
   if( mCommunicator->commRank()==0) {
      char timescalepath[PV_PATH_MAX];
      int chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/%s_timescaleinfo.bin", cpDir, mName);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timescalefile = PV_fopen(timescalepath,"w", mVerifyWrites);
      assert(timescalefile);
      for(int b = 0; b < mBatchWidth; b++){
         if (PV_fwrite(&mTimeScale[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            pvError().printf("HyPerCol::checkpointWrite error writing timeScale to %s\n", timescalefile->name);
         }
         if (PV_fwrite(&mTimeScaleTrue[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            pvError().printf("HyPerCol::checkpointWrite error writing timeScaleTrue to %s\n", timescalefile->name);
         }
         if (PV_fwrite(&mTimeScaleMax[b],1,sizeof(double),timescalefile) != sizeof(double)) {
            pvError().printf("HyPerCol::checkpointWrite error writing timeScaleMax to %s\n", timescalefile->name);
         }
      }
      PV_fclose(timescalefile);
      chars_needed = snprintf(timescalepath, PV_PATH_MAX, "%s/%s_timescaleinfo.txt", cpDir, mName);
      assert(chars_needed < PV_PATH_MAX);
      timescalefile = PV_fopen(timescalepath,"w", mVerifyWrites);
      assert(timescalefile);
      int kb0 = mCommunicator->commBatch() * mBatchWidth;
      for(int b = 0; b < mBatchWidth; b++){
         fprintf(timescalefile->fp,"batch = %d\n", b+kb0);
         fprintf(timescalefile->fp,"time = %g\n", mTimeScale[b]);
         fprintf(timescalefile->fp,"timeScaleTrue = %g\n", mTimeScaleTrue[b]);
      }
      PV_fclose(timescalefile);
   }
   return PV_SUCCESS;
}

std::vector<double> const& AdaptiveTimestepController::calcTimesteps(double timeValue, std::vector<double> const& rawTimeScales) {
   mOldTimeScaleTrue = mTimeScaleTrue;
   mOldTimeScale = mTimeScale;
   mTimeScaleTrue = rawTimeScales;
   for(int b = 0; b < mBatchWidth; b++){
      double E_dt  =  mTimeScaleTrue[b];
      double E_0   =  mOldTimeScaleTrue[b];
      double dE_dt_scaled = (E_0 - E_dt) / mTimeScale[b];

      if ( (dE_dt_scaled <= 0.0) || (E_0 <= 0) || (E_dt <= 0) ) {
         mTimeScale[b]      = mTimeScaleMin;
         mTimeScaleMax[b]   = mTimeScaleMaxBase;
      }
      else {
         double tau_eff_scaled = E_0 / dE_dt_scaled;

         // dt := mTimeScaleMaxBase * tau_eff
         mTimeScale[b] = mChangeTimeScaleMax * tau_eff_scaled;
         mTimeScale[b] = (mTimeScale[b] <= mTimeScaleMax[b]) ? mTimeScale[b] : mTimeScaleMax[b];
         mTimeScale[b] = (mTimeScale[b] <  mTimeScaleMin) ? mTimeScaleMin : mTimeScale[b];

         if (mTimeScale[b] == mTimeScaleMax[b]) {
            mTimeScaleMax[b] = (1 + mChangeTimeScaleMin) * mTimeScaleMax[b];
         }
      }
   }
   return mTimeScale;
}

void AdaptiveTimestepController::writeTimestepInfo(double timeValue, std::ostream& stream) {

   if (mWriteTimeScaleFieldnames) {
      stream << "sim_time = " << timeValue << "\n";
   }
   else {
      stream << timeValue << ", ";
   }
   for(int b = 0; b < mBatchWidth; b++){
      if (mWriteTimeScaleFieldnames) {
         stream << "\tbatch = " << b << ", timeScale = " << mTimeScale[b] << ", " << "timeScaleTrue = " << mTimeScaleTrue[b];
      }
      else {
         stream << b << ", " << mTimeScale[b] << ", " << mTimeScaleTrue[b];
      }
      if (mWriteTimeScaleFieldnames) {
         stream <<  ", " << "timeScaleMax = " << mTimeScaleMax[b] << std::endl;
      }
      else {
         stream <<  ", " << mTimeScaleMax[b] << std::endl;
      }
   }
   stream.flush();
}

} /* namespace PV */
