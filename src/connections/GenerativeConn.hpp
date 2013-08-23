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
   GenerativeConn(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename=NULL, InitWeights *weightInit=NULL);

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename, InitWeights *weightInit);
   virtual int allocateDataStructures();
   virtual int updateWeights(int axonID);
   inline float getRelaxation() { return relaxation; }


protected:
   GenerativeConn();
   virtual int setParams(PVParams * params);
   virtual void readNumAxonalArbors(PVParams * params);
   virtual void readRelaxation(PVParams * params);
   virtual void readNonnegConstraintFlag(PVParams * params);
   virtual void readImprintingFlag(PVParams * params);
   virtual void readWeightDecayFlag(PVParams * params);
   virtual void readWeightDecayRate(PVParams * params);
   virtual void readWeightNoiseLevel(PVParams * params);
#ifdef OBSOLETE // Marked obsolete April 16, 2013.  Implementing the new NormalizeBase class hierarchy
   virtual int initNormalize();
#endif // OBSOLETE
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
