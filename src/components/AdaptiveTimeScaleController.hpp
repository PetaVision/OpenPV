/*
 * AdaptiveTimeScaleController.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESCALECONTROLLER_HPP_
#define ADAPTIVETIMESCALECONTROLLER_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/Checkpointer.hpp"
#include "columns/Communicator.hpp"
#include "io/PrintStream.hpp"
#include <vector>

namespace PV {

class AdaptiveTimeScaleController : public CheckpointerDataInterface {
  public:
   struct TimeScaleInfo {
      std::vector<double> mTimeScale;
      std::vector<double> mTimeScaleMax;
      std::vector<double> mTimeScaleTrue;
   };

   AdaptiveTimeScaleController(
         char const *name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         bool writeTimeScales,
         bool writeTimeScaleFieldnames,
         Communicator *comm,
         bool verifyWrites);
   virtual ~AdaptiveTimeScaleController();
   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) override;
   std::vector<double> const &
   calcTimesteps(double timeValue, std::vector<double> const &rawTimeScales);
   void writeTimestepInfo(double timeValue, PrintStream &stream);

  private:
   void calcTimeScaleTrue(double timeValue);

   // Data members
  protected:
   char *mName;
   int mBatchWidth;
   double mBaseMax;
   double mBaseMin;
   double mTauFactor;
   double mGrowthFactor;
   bool mWriteTimeScales;
   bool mWriteTimeScaleFieldnames;
   Communicator *mCommunicator;
   bool mVerifyWrites;

   TimeScaleInfo mTimeScaleInfo, mOldTimeScaleInfo;
   std::vector<double> mOldTimeScale;
   std::vector<double> mOldTimeScaleTrue;
};

class CheckpointEntryTimeScaleInfo : public CheckpointEntry {
  public:
   CheckpointEntryTimeScaleInfo(
         std::string const &name,
         Communicator *communicator,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(name, communicator), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   CheckpointEntryTimeScaleInfo(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(objName, dataName, communicator), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   AdaptiveTimeScaleController::TimeScaleInfo *mTimeScaleInfoPtr;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALECONTROLLER_HPP_ */
