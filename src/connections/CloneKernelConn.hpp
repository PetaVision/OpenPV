/*
 * CloneKernelConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONN_HPP_
#define CLONEKERNELCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class CloneKernelConn : public KernelConn {

public:
   CloneKernelConn();
   CloneKernelConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn);
   virtual ~CloneKernelConn();
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      KernelConn * originalConn);

   virtual int setPatchSize(const char * filename);
   // filename should always be null, but this prototype is needed because
   // the inherited method is called by the base class's initialize.

   virtual int initNormalize();

protected:
   PVPatch ** allocWeights(PVPatch ** patches, int nPatches,
         int nxPatch, int nyPatch, int nfPatch, int axonId);
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
            const char * filename);
   int deleteWeights();

   KernelConn * originalConn;

}; // end class CloneKernelConn

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
