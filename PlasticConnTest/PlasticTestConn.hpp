/*
 * PlasticTestConn.hpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#ifndef PLASTICTESTCONN_HPP_
#define PLASTICTESTCONN_HPP_

#include "../PetaVision/src/connections/KernelConn.hpp"

namespace PV {

class PlasticTestConn : public KernelConn {
public:
	PlasticTestConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
		      const char * filename, InitWeights *weightInit);
	virtual ~PlasticTestConn();
protected:
	virtual int update_dW(int axonId);
	virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
};

} /* namespace PV */
#endif /* PLASTICTESTCONN_HPP_ */
