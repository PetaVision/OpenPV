/*
 * ImprintConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef IMPRINTCONN_HPP_
#define IMPRINTCONN_HPP_

#include "HyPerConn.hpp"
#include "PlasticCloneConn.hpp"
namespace PV {

class ImprintConn: public HyPerConn {

public:
   ImprintConn();
   ImprintConn(const char * name, HyPerCol * hc);
   virtual ~ImprintConn();

   virtual int allocateDataStructures();

   virtual int update_dW(int arbor_ID);
   virtual int updateWeights(int arbor_ID);
   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int checkpointWrite(const char * cpDir);

protected:
   //int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_imprintTimeThresh(enum ParamsIOFlag ioFlag);
   //virtual void ioParam_imprintChance(enum ParamsIOFlag ioFlag);
   int imprintFeature(int arborId, int kExt);
   double imprintTimeThresh;
   double* lastActiveTime;

private:
   int initialize_base();
   bool * imprinted;
   float imprintChance;

}; // end class 

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
