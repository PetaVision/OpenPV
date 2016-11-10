/*
 * MomentumConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMCONN_HPP_
#define MOMENTUMCONN_HPP_

#include "HyPerConn.hpp"
namespace PV {

class MomentumConn : public HyPerConn {

  public:
   MomentumConn();
   MomentumConn(const char *name, HyPerCol *hc);
   virtual ~MomentumConn();
   virtual int allocateDataStructures();

   virtual int applyMomentum(int arbor_ID);

  protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchPeriod(enum ParamsIOFlag ioFlag);

   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) override;

   inline pvwdata_t *get_prev_dwDataHead(int arborId, int dataIndex) {
      return &prev_dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   virtual int calc_dW();
   virtual int updateWeights(int arborId);

  private:
   int initialize_base();
   pvwdata_t **prev_dwDataStart;
   float momentumTau;
   float momentumDecay;
   char *momentumMethod;
   int timeBatchIdx;
   int timeBatchPeriod;

}; // end class MomentumConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
