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
   ImprintConn(
         const char *name,
         HyPerCol *hc,
         InitWeights *weightInitializer  = NULL,
         NormalizeBase *weightNormalizer = NULL);
   virtual ~ImprintConn();

   virtual int allocateDataStructures();

  protected:
   virtual int initialize_dW(int arborId);
   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) override;
   virtual int update_dW(int arbor_ID);
   virtual int updateWeights(int arbor_ID);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
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
