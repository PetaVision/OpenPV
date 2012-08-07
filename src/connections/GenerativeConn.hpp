/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "KernelConn.hpp"
#include "../utils/pv_random.h"

namespace PV {

class GenerativeConn : public KernelConn {
public:
   GenerativeConn(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         const char * filename=NULL, InitWeights *weightInit=NULL);

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         const char * filename, InitWeights *weightInit);
#ifdef OBSOLETE
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
#endif // OBSOLETE
   inline float getRelaxation() { return relaxation; }
   virtual int updateWeights(int axonID);
   virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);


protected:
   GenerativeConn();
   virtual int initNormalize();
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
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
