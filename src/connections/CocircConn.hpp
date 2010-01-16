/*
 * CocircConn.h
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#ifndef COCIRCCONN_HPP_
#define COCIRCCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class CocircConn: public KernelConn {
private:

public:

   CocircConn();
   CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         int channel, const char * filename);
   CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         int channel);
   CocircConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeCocircWeights(PVPatch ** patches, int numPatches);
   int cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float deltaThetaMax, float cocirc_self,
         float dKv, int numFlanks, float shift, float aspect, float rotate, float sigma,
         float r2Max, float strength);

};

}

#endif /* COCIRCCONN_HPP_ */
