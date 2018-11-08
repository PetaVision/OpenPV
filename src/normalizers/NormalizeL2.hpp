/*
 * NormalizeL2.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEL2_HPP_
#define NORMALIZEL2_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeL2 : public NormalizeMultiply {
   // Member functions
  public:
   NormalizeL2(const char *name, PVParams *params, Communicator *comm);
   virtual ~NormalizeL2();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeL2();
   void initialize(const char *name, PVParams *params, Communicator *comm);

   virtual void ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

   // Member variables
  protected:
   float minL2NormTolerated; // Error if sqrt(sum(weights^2)) in any patch is less than this amount.
}; // class NormalizeL2

} /* namespace PV */
#endif /* NORMALIZEL2_HPP_ */
