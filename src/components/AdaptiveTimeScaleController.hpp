/*
 * AdaptiveTimeScaleController.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESCALECONTROLLER_HPP_
#define ADAPTIVETIMESCALECONTROLLER_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/CheckpointerDataInterface.hpp"
#include "columns/Communicator.hpp"
#include "io/PrintStream.hpp"
#include "structures/MPIBlock.hpp"
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
         bool writeTimeScaleFieldnames,
         Communicator *comm);
   virtual ~AdaptiveTimeScaleController();
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual std::vector<double>
   calcTimesteps(double timeValue, std::vector<double> const &rawTimeScales);
   void writeTimestepInfo(double timeValue, std::vector<PrintStream *> &streams);

   // Data members
  protected:
   char *mName;
   int mBatchWidth;
   double mBaseMax;
   double mBaseMin;
   double mTauFactor;
   double mGrowthFactor;
   bool mWriteTimeScaleFieldnames;
   Communicator *mCommunicator;

   TimeScaleInfo mTimeScaleInfo, mOldTimeScaleInfo;
   std::vector<double> mOldTimeScale;
   std::vector<double> mOldTimeScaleTrue;
};

class CheckpointEntryTimeScaleInfo : public CheckpointEntry {
  public:
   CheckpointEntryTimeScaleInfo(
         std::string const &name,
         MPIBlock const *mpiBlock,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(name, mpiBlock), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   CheckpointEntryTimeScaleInfo(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(objName, dataName, mpiBlock), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   AdaptiveTimeScaleController::TimeScaleInfo *mTimeScaleInfoPtr;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALECONTROLLER_HPP_ */
