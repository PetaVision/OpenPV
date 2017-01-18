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

   inline float const *get_prev_dwDataStart(int arborId) { return prev_dwDataStart[arborId]; }
   inline char const *getMomentumMethod() { return momentumMethod; }

  protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);

   // batchPeriod was marked obsolete Jan 17, 2017.
   /**
    * batchPeriod is obsolete. Use HyPerCol nbatch parameter instead.
    */
   virtual void ioParam_batchPeriod(enum ParamsIOFlag ioFlag);

   virtual int registerData(Checkpointer *checkpointer, std::string const &objName) override;
   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;

   inline float *get_prev_dwDataHead(int arborId, int dataIndex) {
      return &prev_dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   virtual int updateWeights(int arborId);

  private:
   int initialize_base();
   float **prev_dwDataStart;
   float momentumTau;
   float momentumDecay;
   char *momentumMethod;

}; // end class MomentumConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
