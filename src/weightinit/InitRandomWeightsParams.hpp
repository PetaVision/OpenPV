/*
 * InitRandomWeightsParams.hpp
 *
 *  Created on: Feb 26, 2014
 *      Author: pschultz
 */

#ifndef INITRANDOMWEIGHTSPARAMS_HPP_
#define INITRANDOMWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitRandomWeightsParams: public PV::InitWeightsParams {
public:
   InitRandomWeightsParams(HyPerConn * pConn);
   virtual ~InitRandomWeightsParams();

protected:
   InitRandomWeightsParams();
};

} /* namespace PV */
#endif /* INITRANDOMWEIGHTSPARAMS_HPP_ */
