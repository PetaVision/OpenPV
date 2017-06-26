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

class ImprintConn : public HyPerConn {
  public:
   ImprintConn();
   ImprintConn(const char *name, HyPerCol *hc);
   virtual ~ImprintConn();

   virtual int allocateDataStructures() override;

  protected:
   virtual int initialize_dW(int arborId) override;
   virtual int registerData(Checkpointer *checkpointer) override;
   virtual int update_dW(int arborID) override;
   virtual int updateWeights(int arbor_ID) override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_imprintTimeThresh(enum ParamsIOFlag ioFlag);
   int imprintFeature(int arborId, int batchId, int kExt);
   double imprintTimeThresh;
   double *lastActiveTime;

  private:
   int initialize_base();
   bool *imprinted;
   float imprintChance;

}; // end class ImprintConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
