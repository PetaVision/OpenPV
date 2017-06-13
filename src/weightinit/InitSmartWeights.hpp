/*
 * InitSmartWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITSMARTWEIGHTS_HPP_
#define INITSMARTWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitSmartWeights : public PV::InitWeights {
  public:
   InitSmartWeights(char const *name, HyPerCol *hc);
   InitSmartWeights();
   virtual ~InitSmartWeights();

   virtual void calcWeights(float *dataStart, int patchIndex, int arborId) override;

  protected:
   int initialize(char const *name, HyPerCol *hc);

  private:
   int initialize_base();
   void smartWeights(float *dataStart, int k);
}; // class InitSmartWeights

} /* namespace PV */
#endif /* INITSMARTWEIGHTS_HPP_ */
