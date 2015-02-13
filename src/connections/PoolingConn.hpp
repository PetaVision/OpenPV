/*
 * PoolingConn.hpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class PoolingConn : public HyPerConn {
public:
   PoolingConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~PoolingConn();

   HyPerLayer * getPre2() { return pre2; }
   HyPerLayer * getPost2() { return post2; }

   virtual int communicateInitInfo();
   int updateWeights(int axonID);

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_secondaryPreLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_secondaryPostLayerName(enum ParamsIOFlag ioFlag);
   bool checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2);

   char * preLayerName2;
   char * postLayerName2;
   HyPerLayer * pre2;
   HyPerLayer * post2;
};  // end class PoolingConn

}  // end namespace PV

#endif /* POOLINGCONN_HPP_ */
