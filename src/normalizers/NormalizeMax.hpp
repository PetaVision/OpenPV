/*
 * NormalizeMax.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEMAX_HPP_
#define NORMALIZEMAX_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeMax : public NormalizeMultiply {
   // Member functions
  public:
   NormalizeMax(const char *name, PVParams *params, Communicator *comm);
   virtual ~NormalizeMax();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeMax();
   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual void ioParam_minMaxTolerated(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

   // Member variables
  protected:
   float minMaxTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

} /* namespace PV */
#endif /* NORMALIZEMAX_HPP_ */
