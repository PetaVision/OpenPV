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
#include "probes/ProbeData.hpp"
#include "structures/MPIBlock.hpp"
#include <memory>
#include <vector>

namespace PV {

struct TimeScaleData {
   double mTimeScale;
   double mTimeScaleMax;
   double mTimeScaleTrue;
};

class AdaptiveTimeScaleController : public CheckpointerDataInterface {
  public:
   AdaptiveTimeScaleController(
         char const *name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         Communicator const *comm);
   virtual ~AdaptiveTimeScaleController();
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual std::vector<TimeScaleData> const &calcTimesteps(std::vector<double> const &timeScales);

   // Data members
  protected:
   char *mName;
   int mBatchWidth;
   double mBaseMax;
   double mBaseMin;
   double mTauFactor;
   double mGrowthFactor;
   bool mWriteTimeScaleFieldnames;
   Communicator const *mCommunicator;

   std::vector<TimeScaleData> mTimeScaleInfo, mOldTimeScaleInfo;
};

class CheckpointEntryTimeScaleInfo : public CheckpointEntry {
  public:
   CheckpointEntryTimeScaleInfo(
         std::string const &name,
         TimeScaleData *timeScaleDataPtr,
         int batchSize)
         : CheckpointEntry(name), mTimeScaleDataPtr(timeScaleDataPtr), mBatchSize(batchSize) {}
   CheckpointEntryTimeScaleInfo(
         std::string const &objName,
         std::string const &dataName,
         TimeScaleData *timeScaleDataPtr,
         int batchSize)
         : CheckpointEntry(objName, dataName), mTimeScaleDataPtr(timeScaleDataPtr), mBatchSize(batchSize) {}
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(
         std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   TimeScaleData *mTimeScaleDataPtr;
   int mBatchSize;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALECONTROLLER_HPP_ */
