/*
 * CopyConn.hpp
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#ifndef COPYCONN_HPP_
#define COPYCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class CopyConn : public HyPerConn {
  public:
   CopyConn(char const *name, HyPerCol *hc);
   virtual ~CopyConn();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual bool needUpdate(double time, double dt) override;
   virtual int updateState(double time, double dt) override;
   char const *getOriginalConnName() { return originalConnName; }
   HyPerConn *getOriginalConn() { return originalConn; }

  protected:
   CopyConn();
   int initialize(char const *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   /**
    * List of parameters needed from the CopyConn class
    * @name CopyConn Parameters
    * @{
    */

   /**
    * @brief CopyConn inherits sharedWeights from the original connection, instead of reading it
    * from parameters
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief weightInitType is not used by CopyConn.
    */
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn inherits nxp from the original connection, instead of reading it from
    * parameters
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn inherits nyp from the original connection, instead of reading it from
    * parameters
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn inherits nfp from the original connection, instead of reading it from
    * parameters
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief initializeFromCheckpointFlag is not used by CopyConn.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn inherits numAxonalArbors from the original connection, instead of reading it
    * from parameters
    */
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn inherits plasticityFlag from the original connection, instead of reading it
    * from parameters
    */
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief CopyConn does not use trigger layers
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief weightUpdatePeriod is not used by CopyConn.
    */
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) override;

   /**
     * @brief initialWeightUpdateTime is not used by CopyConn.
     */
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief dWMax is not used by CopyConn.
    */
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief useMask is not used by CopyConn.
    */
   virtual void ioParam_useMask(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief maskLayerName is not used by CopyConn.
    */
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief originalConnName (required): The name of the connection the weights will be copied from
    */
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   /** @} */

   virtual int setPatchSize() override;

   virtual int setInitialValues() override;
   virtual PVPatch ***initializeWeights(PVPatch ***arbors, float **dataStart) override;

   virtual int updateWeights(int arborId = 0) override;
   int copy(int arborId = 0);

   char *originalConnName;
   HyPerConn *originalConn;

  private:
   int initialize_base();
}; // end class CopyConn

} /* namespace PV */

#endif /* COPYCONN_HPP_ */
