/*
 * AdaptiveTimestepController.hpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#ifndef ADAPTIVETIMESTEPCONTROLLER_HPP_
#define ADAPTIVETIMESTEPCONTROLLER_HPP_

#include "columns/Communicator.hpp"
#include <ostream>
#include <vector>

namespace PV {

class AdaptiveTimestepController {
public:
   AdaptiveTimestepController(
         char const * name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         bool writeTimescales,
         bool writeTimeScaleFieldnames,
         Communicator * comm,
         bool verifyWrites);
   virtual ~AdaptiveTimestepController();
   virtual int checkpointRead(const char * cpDir, double * timeptr);
   virtual int checkpointWrite(const char * cpDir);
   std::vector<double> const& calcTimesteps(double timeValue, std::vector<double> const& rawTimeScales);
   void writeTimestepInfo(double timeValue, std::ostream& stream);

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
   bool   mWriteTimescales;
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

#endif /* ADAPTIVETIMESTEPCONTROLLER_HPP_ */
