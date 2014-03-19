/*
 * NormalizeGap.hpp
 *
 *  Created on: Feb 28, 2014
 *      Author: pschultz
 *
 *  A weight-normalization class to be used by GapConns.
 *  GapConn requires weights be normalized so that the sum of all weights
 *  going into a given postsynaptic neuron is a constant (the strength
 *  parameter).   Thus, NormalizeGap is a derived class of NormalizeSum with
 *  normalizeFromPostPerspective set to true.
 */

#ifndef NORMALIZEGAP_HPP_
#define NORMALIZEGAP_HPP_

#include "NormalizeSum.hpp"
#include "../connections/GapConn.hpp"

namespace PV {

class NormalizeGap: public PV::NormalizeSum {
public:
   NormalizeGap(GapConn * callingConn);
   virtual ~NormalizeGap();

protected:
   NormalizeGap();
   int initialize(GapConn * callingConn);
   virtual void ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
};

} /* namespace PV */
#endif /* NORMALIZEGAP_HPP_ */
