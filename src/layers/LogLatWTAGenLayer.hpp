/*
 * LogLatWTAGenLayer.hpp
 *
 * A subclass of GenerativeLayer that
 * uses log(lateral winner-take-all) dynamics for sparsity
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#ifndef LOGLATWTAGENLAYER_HPP_
#define LOGLATWTAGENLAYER_HPP_

#include "GenerativeLayer.hpp"

namespace PV {

class LogLatWTAGenLayer : public GenerativeLayer {
public:
   LogLatWTAGenLayer(const char * name, HyPerCol * hc);
   ~LogLatWTAGenLayer();

protected:
   LogLatWTAGenLayer();
   int updateSparsityTermDerivative();
   virtual pvdata_t latWTAterm(pvdata_t * V, int nf);
private:
   int initialize_base();
};

}  // end namespace PV block

#endif /* LOGLATWTAGENLAYER_HPP_ */
