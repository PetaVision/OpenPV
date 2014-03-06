/*
 * ImprintConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef IMPRINTCONN_HPP_
#define IMPRINTCONN_HPP_

#include "KernelConn.hpp"
namespace PV {

class ImprintConn: public KernelConn {

public:
   ImprintConn();
   ImprintConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit);
   virtual ~ImprintConn();

   //virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   // virtual int setPatchSize(const char * filename); // Now a protected method.

   virtual int update_dW(int arbor_ID);
   //virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
   //virtual int updateWeights(int arbor_ID);

protected:
   //int initialize(const char * name, HyPerCol * hc,
   //               const char * pre_layer_name, const char * post_layer_name,
   //               const char * filename, InitWeights *weightInit=NULL);
   virtual int setParams(PVParams * params);
   bool imprintFeature(int arborId, int kExt);
   double imprintTimeThresh;
   double* lastActiveTime;

private:
   int initialize_base();
   bool * imprinted;
   //bool allDidImprint;

}; // end class 

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
