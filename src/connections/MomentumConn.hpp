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
  protected:
   /**
    * @brief momentumMethod: The momentum method to use
    * @details Assuming a = dwMax * pre * post
    * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
    * viscosity: deltaW(t) = (deltaW(t-1) * exp(-1/momentumTau)) + a
    * alex: deltaW(t) = momentumTau * delta(t-1) - momentumDecay * dwMax * w(t) + a
    */
   virtual void ioParam_momentumMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumTau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_momentumDecay(enum ParamsIOFlag ioFlag);

   // batchPeriod was marked obsolete Jan 17, 2017.
   /**
    * batchPeriod is obsolete. Use HyPerCol nbatch parameter instead.
    */
   virtual void ioParam_batchPeriod(enum ParamsIOFlag ioFlag);

  public:
   MomentumConn();
   MomentumConn(const char *name, HyPerCol *hc);
   virtual ~MomentumConn();
   virtual int allocateDataStructures() override;

   inline float const *get_prev_dwDataStart(int arborId) { return prev_dwDataStart[arborId]; }
   inline char const *getMomentumMethod() { return momentumMethod; }

  protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int registerData(Checkpointer *checkpointer) override;
   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;

   void applyMomentum(int arbor_ID);
   void applyMomentum(int arbor_ID, float dwFactor, float wFactor);

   inline float *get_prev_dwDataHead(int arborId, int dataIndex) {
      return &prev_dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   virtual int updateWeights(int arborId) override;

  private:
   enum Method { SIMPLE, VISCOSITY, ALEX };

   int initialize_base();

   float **prev_dwDataStart;
   char *momentumMethod;
   float momentumTau;
   float momentumDecay;
   Method method;

}; // end class MomentumConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
