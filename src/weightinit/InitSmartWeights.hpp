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

class InitSmartWeights : public InitWeights {
  public:
   InitSmartWeights(char const *name, PVParams *params, Communicator const *comm);
   InitSmartWeights();
   virtual ~InitSmartWeights();

   virtual void calcWeights(int patchIndex, int arborId) override;

  protected:
   void initialize(char const *name, PVParams *params, Communicator const *comm);

  private:
   void smartWeights(float *dataStart, int k);
}; // class InitSmartWeights

} /* namespace PV */
#endif /* INITSMARTWEIGHTS_HPP_ */
