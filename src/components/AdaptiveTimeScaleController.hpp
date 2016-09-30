/*
 * AdaptiveTimeScaleController.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESCALECONTROLLER_HPP_
#define ADAPTIVETIMESCALECONTROLLER_HPP_

#include "columns/Communicator.hpp"
#include "io/PrintStream.hpp"
#include <vector>

namespace PV {

class AdaptiveTimeScaleController {
public:
   AdaptiveTimeScaleController(
         char const * name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         bool writeTimeScales,
         bool writeTimeScaleFieldnames,
         Communicator * comm,
         bool verifyWrites);
   virtual ~AdaptiveTimeScaleController();
   virtual int checkpointRead(const char * cpDir, double * timeptr);
   virtual int checkpointWrite(const char * cpDir);
   std::vector<double> const& calcTimesteps(double timeValue, std::vector<double> const& rawTimeScales);
   void writeTimestepInfo(double timeValue, PrintStream &stream);

private:
   void calcTimeScaleTrue(double timeValue);

// Data members
protected:
   char * mName;
   int mBatchWidth;
   double mBaseMax;
   double mBaseMin;
   double mTauFactor;
   double mGrowthFactor;
   bool   mWriteTimeScales;
   bool   mWriteTimeScaleFieldnames;
   Communicator * mCommunicator;
   bool mVerifyWrites;

   std::vector<double> mTimeScale;
   std::vector<double> mTimeScaleMax;
   std::vector<double> mTimeScaleTrue;
   std::vector<double> mOldTimeScale;
   std::vector<double> mOldTimeScaleTrue;
};

} /* namespace PV */

#endif /* ADAPTIVETIMESCALECONTROLLER_HPP_ */
