/*
 * StochasticReleaseTestProbe.hpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#ifndef STOCHASTICRELEASETESTPROBE_HPP_
#define STOCHASTICRELEASETESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/buildandrun.hpp>
#include <math.h>
#include <stdlib.h>

namespace PV {

class StochasticReleaseTestProbe : public PV::StatsProbe{
public:
   StochasticReleaseTestProbe(const char * name, HyPerCol * hc);
   virtual ~StochasticReleaseTestProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timed);

protected:
   StochasticReleaseTestProbe();
   int initStochasticReleaseTestProbe(const char * name, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   int computePValues(long int step, int f);
private:
   int initialize_base();

// Member variables
protected:
   HyPerConn * conn; // The connection for which targetLayer is the postsynaptic layer.  There must be exactly one such conn.
   double * pvalues;      // The two-tailed p-value of the nnz value of each timestep.
}; // end class StochasticReleaseTestProbe

BaseObject * createStochasticReleaseTestProbe(const char * name, HyPerCol * hc);

} /* namespace PV */
#endif /* STOCHASTICRELEASETESTPROBE_HPP_ */
