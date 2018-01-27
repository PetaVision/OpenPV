/*
 * NormalizeNone.hpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#ifndef NORMALIZENONE_HPP_
#define NORMALIZENONE_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeNone : public NormalizeBase {
   // Member functions
  protected:
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag) override {}
   virtual void ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) override {}
   virtual void ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) override {}
   virtual void ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) override {}

  public:
   NormalizeNone(const char *name, HyPerCol *hc);
   virtual ~NormalizeNone();

  protected:
   NormalizeNone();
   int initialize(const char *name, HyPerCol *hc);
}; // class NormalizeNone

} /* namespace PV */

#endif /* NORMALIZENONE_HPP_ */
