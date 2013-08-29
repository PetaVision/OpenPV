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

namespace PV {

class StochasticReleaseTestProbe : public PV::StatsProbe{
public:
   StochasticReleaseTestProbe(const char * name, HyPerCol * hc);
   virtual ~StochasticReleaseTestProbe();

   virtual int outputState(double timed);

protected:
   StochasticReleaseTestProbe();
   int initStochasticReleaseTestProbe(const char * name, HyPerCol * hc);

private:
   int initialize_base();

// Member variables
protected:
   HyPerConn * conn; // The connection for which targetLayer is the postsynaptic layer.  There must be exactly one such conn.
   int bins[9]; // Generate a histogram of deviations from expected value.
   // nnz is random with mean a*N and standard deviation sqrt(a*(1-a)*N)
   // where N is the number of neurons and a is the presynaptic activity, truncated to lie in the interval [0,1].
   // bins[0] is the count of the times nnz < mean - 3.5*sigma
   // bins[1] is the count of the times (nnz-mean)/sigma is in [-3.5,-2.5)
   // bins[2]: (nnz-mean)/sigma is in [-2.5,-1.5)
   // bins[3]: [-1.5,-0.5)
   // bins[4]: [-0.5,0.5]
   // bins[5]: (0.5,1.5]
   // etc.
   // bins[8] (nnz-mean)/sigma > 3.5
   int sumbins;
   double binprobs[9]; // Probability that a given trial should end up in a given bin
};

} /* namespace PV */
#endif /* STOCHASTICRELEASETESTPROBE_HPP_ */
