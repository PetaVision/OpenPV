/*
 * KernelConn.hpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#ifndef KERNELCONN_HPP_
#define KERNELCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class KernelConn: public HyPerConn {

private:
   KernelConn();

public:

   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              int channel, const char * filename);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              int channel);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   virtual int numDataPatches(int arbor);

protected:
   PVPatch      ** kernelPatches;   // list of weight patches
   virtual int deleteWeights();
   virtual int initialize_base();
   virtual PVPatch **  allocWeights(PVPatch ** patches);
   virtual PVPatch ** createWeights(PVPatch ** patches);
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
         const char * filename);
   virtual int writeWeights(const char * filename, float time);

};

}

#endif /* KERNELCONN_HPP_ */
