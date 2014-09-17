/*
 * VaryingHyPerConn.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef VARYINGHYPERCONN_HPP_
#define VARYINGHYPERCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class VaryingHyPerConn : public HyPerConn {

public:
   VaryingHyPerConn(const char * name, HyPerCol * hc);
   virtual ~VaryingHyPerConn();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonId = 0);

protected:
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void readPlasticityFlag(PVParams * inputParams);
   virtual int calc_dW(int axonId);

}; // end class VaryingHyPerConn

}  // end namespace PV block


#endif /* VARYINGHYPERCONN_HPP_ */
