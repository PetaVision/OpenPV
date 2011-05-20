/*
 * PtwiseProductLayer.hpp
 *
 * The output V is the pointwise product of phiExc and phiInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEPRODUCTLAYER_HPP_
#define PTWISEPRODUCTLAYER_HPP_

#include "../PetaVision/src/layers/ANNLayer.hpp"

namespace PV {

class PtwiseProductLayer : public ANNLayer {
public:
	PtwiseProductLayer(const char * name, HyPerCol * hc);

	virtual int updateV();
};  // end class PtwiseProductLayer

}  // end namespace PV

#endif /* PTWISEPRODUCTLAYER_HPP_ */
