/*
 * PlasticCloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef PLASTICCLONECONN_HPP_
#define PLASTICCLONECONN_HPP_

#include "CloneConn.hpp"

namespace PV {

class PlasticCloneConn : public CloneConn {
  protected:
   /**
    * List of parameters needed from the PlasticCloneConn class
    * @name PlasticCloneConn Parameters
    * @{
    */

   /**
    * @brief plasticityFlag: PlasticCloneConn always sets plasticityFlag to true.
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief triggerLayerName: PlasticCloneConn does not use triggerLayerName,
    * since updating weights is managed by the original conn.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief triggerOffset: PlasticCloneConn does not use triggerOffset,
    * since updating weights is managed by the original conn.
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief weightUpdatePeriod: PlasticCloneConn does not do weight updates,
    * since its weights are updated by the original conn.
    */
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief initialWeightUpdateTime: PlasticCloneConn does not do weight
    * updates, since its weights are updated by the original conn.
    */
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief immediateWeightUpdate: PlasticCloneConn does not do weight updates,
    * since its weights are updated by the original conn.
    */
   virtual void ioParam_immediateWeightUpdate(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief dWMax: PlasticCloneConn uses the same dWMax as the original conn.
    */
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief dWMaxDecayInterval: PlasticCloneConn uses the same dWMax as the
    * original conn.
    */
   virtual void ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief dWMaxDecayFactor: PlasticCloneConn uses the same dWMax as the
    * original conn.
    */
   virtual void ioParam_dWMaxDecayFactor(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief normalizeDw: PlasticCloneConn does not do weight updates,
    * since its weights are updated by the original conn.
    */
   virtual void ioParam_normalizeDw(enum ParamsIOFlag ioFlag) override;
   /** @} */

  public:
   PlasticCloneConn(const char *name, HyPerCol *hc);
   virtual ~PlasticCloneConn();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   PlasticCloneConn();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int cloneParameters() override;
   virtual int constructWeights() override;
   int deleteWeights();

  private:
   int initialize_base();

}; // end class PlasticCloneConn

} // end namespace PV

#endif /* CLONECONN_HPP_ */
