/*
 * PoolingGenConn.hpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef POOLINGGENCONN_HPP_
#define POOLINGGENCONN_HPP_

#include "GenerativeConn.hpp"

namespace PV {

class PoolingGenConn : public GenerativeConn {
public:
   PoolingGenConn(const char * name, HyPerCol * hc);
   virtual ~PoolingGenConn();

   HyPerLayer * getPre2() { return pre2; }
   HyPerLayer * getPost2() { return post2; }

   virtual int communicateInitInfo();
   int updateWeights(int axonID);

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_secondaryPreLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_secondaryPostLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessPre(enum ParamsIOFlag ioFlag);
   virtual void ioParam_slownessPost(enum ParamsIOFlag ioFlag);
   bool checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2);
   int getSlownessLayerName(char ** l, const char * paramname);

   char * preLayerName2;
   char * postLayerName2;
   char * slownessPreName;
   char * slownessPostName;
   HyPerLayer * pre2;
   HyPerLayer * post2;
   bool slownessFlag;
   HyPerLayer * slownessPre;
   HyPerLayer * slownessPost;
};  // end class PoolingGenConn

}  // end namespace PV

#endif /* GENPOOLCONN_HPP_ */
