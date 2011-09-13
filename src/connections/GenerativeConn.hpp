/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class GenerativeConn : public KernelConn {
public:
   GenerativeConn();
   GenerativeConn(const char * name, HyPerCol * hc,
       HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
   GenerativeConn(const char * name, HyPerCol * hc,
       HyPerLayer * pre, HyPerLayer * post, ChannelType channel, InitWeights *weightInit);
   GenerativeConn(const char * name, HyPerCol * hc,
       HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
           const char * filename);
   GenerativeConn(const char * name, HyPerCol * hc,
       HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
           const char * filename, InitWeights *weightInit);

   int initialize_base();
//    int initialize(const char * name, HyPerCol * hc,
//            HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
//            const char * filename=NULL);
   int initialize(const char * name, HyPerCol * hc,
           HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
           const char * filename, InitWeights *weightInit);
#ifdef OBSOLETE
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
#endif // OBSOLETE
   inline float getRelaxation() { return relaxation; }
   virtual int calc_dW(int axonID);
   virtual int updateWeights(int axonID);
   virtual int initNormalize();
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);


protected:
   float relaxation;
   PVPatch ** dWPatches;
   int * patchindices; // An array whose length is the number of extended neurons in the presynaptic layer
   bool nonnegConstraintFlag;
   int normalizeMethod;
   float normalizeConstant;
#ifdef PV_USE_MPI
   virtual int reduceKernels(int axonID);
#endif // PV_USE_MPI
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
