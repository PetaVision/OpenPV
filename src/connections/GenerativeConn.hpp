/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "HyPerConn.hpp"
#include "../columns/Random.hpp"

namespace PV {

class GenerativeConn : public HyPerConn {
public:
   GenerativeConn(const char * name, HyPerCol * hc);

   int initialize_base();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonID);


protected:
   GenerativeConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nonnegConstraintFlag(enum ParamsIOFlag ioFlag);
   virtual int update_dW(int axonID);

   bool nonnegConstraintFlag;
   float normalizeConstant;
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
