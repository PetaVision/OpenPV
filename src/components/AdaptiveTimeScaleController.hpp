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
#include <memory>
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
         Communicator const *comm);
   virtual ~AdaptiveTimeScaleController();
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
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
   Communicator const *mCommunicator;

   TimeScaleInfo mTimeScaleInfo, mOldTimeScaleInfo;
   std::vector<double> mOldTimeScale;
   std::vector<double> mOldTimeScaleTrue;
};

class CheckpointEntryTimeScaleInfo : public CheckpointEntry {
  public:
   CheckpointEntryTimeScaleInfo(
         std::string const &name,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(name), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   CheckpointEntryTimeScaleInfo(
         std::string const &objName,
         std::string const &dataName,
         AdaptiveTimeScaleController::TimeScaleInfo *timeScaleInfoPtr)
         : CheckpointEntry(objName, dataName), mTimeScaleInfoPtr(timeScaleInfoPtr) {}
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(
         std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   AdaptiveTimeScaleController::TimeScaleInfo *mTimeScaleInfoPtr;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALECONTROLLER_HPP_ */
