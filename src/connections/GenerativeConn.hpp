/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "KernelConn.hpp"
#include "../columns/Random.hpp"

namespace PV {

class GenerativeConn : public KernelConn {
public:
   GenerativeConn(const char * name, HyPerCol * hc);

   int initialize_base();
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonID);
   inline float getRelaxation() { return relaxation; }


protected:
   GenerativeConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_relaxation(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nonnegConstraintFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imprintingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightDecayFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightDecayRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightNoiseLevel(enum ParamsIOFlag ioFlag);
   virtual int update_dW(int axonID);

   float relaxation;
   bool nonnegConstraintFlag;
   int normalizeMethod;
   float normalizeConstant;
   bool imprintingFlag;
   int imprintCount;
   bool weightDecayFlag;  // Include Nugent-like decay and noise on weights.  If flag is set, use weightDecayRate and weightNoiseLevel
   float weightDecayRate; // Include a term of weightDecayRate * W_{ij} in dW_{ij}
   float weightNoiseLevel;// Include a random fluctuation term, uniformly distributed on [-weightNoiseLevel,weightNoiseLevel], in dW_{ij}
   Random * noise; // Random number generator for noise
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
