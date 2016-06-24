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
   VaryingHyPerConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~VaryingHyPerConn();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonId = 0);

protected:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int calc_dW(int axonId);

}; // end class VaryingHyPerConn

BaseObject * createVaryingHyPerConn(char const * name, HyPerCol * hc);

}  // end namespace PV block


#endif /* VARYINGHYPERCONN_HPP_ */
