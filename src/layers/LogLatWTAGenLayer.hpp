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
   virtual ~LogLatWTAGenLayer();

   virtual int updateState(float timef, float dt);

protected:
   LogLatWTAGenLayer();
   /* static */ int updateState(float timef, float dt, int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynAux, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence);
   // int updateSparsityTermDerivative();
   // virtual pvdata_t latWTAterm(pvdata_t * V, int nf);
private:
   int initialize_base();
};

}  // end namespace PV block

#endif /* LOGLATWTAGENLAYER_HPP_ */
