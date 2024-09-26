/*
 * PatchSize.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#ifndef PATCHSIZE_HPP_
#define PATCHSIZE_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

/**
 * A component to contain the x-by-y-by-f dimensions of a HyPerConn's or PoolingConn's
 * patch size. The dimensions are read from the parameters nxp, nyp, and nfp. They are
 * retrieved using the getPatchSizeX(), getPatchSizeY(), and getPatchSizeF() methods.
 *
 */
class PatchSize : public BaseObject {
  protected:
   /**
    * List of parameters needed from the PatchSize class
    * @name PatchSize Parameters
    * @{
    */

   /**
    * @brief nxp: Specifies the x patch size
    * @details If one pre to many post, nxp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nxp restricted to an odd number
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nyp: Specifies the y patch size
    * @details If one pre to many post, nyp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nyp restricted to an odd number
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nfp: Specifies the post feature patch size. If negative, it can be
    * set during the CommunicateInitInfo phase.
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);

   /** @} */ // end of PatchSize parameters

  public:
   PatchSize(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PatchSize();

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   ConnectionData *getConnectionData() const { return mConnectionData; }

   /**
    * Calculates the patch size from the postsynaptic perspective, given the patch size from the
    * presynaptic perspective and the PVLayerLoc structs for the pre- and post-synaptic layers.
    *
    * If numNeuronsPre == numNeuronsPost, the return value is prePatchSize.
    *
    * If numNeuronsPre > numNeuronsPost, numNeuronsPre must be an integer multiple of
    * numNeuronsPost. The return value is prePatchSize * (numNeuronsPre / numNeuronsPost);
    *
    * If numNeuronsPre < numNeuronsPost, numNeuronsPost must be an integer multiple of
    * numNeuronsPre, and prePatchSize must be in integer multiple of their quotient.
    * The return value is the prePatchSize / (numNeuronsPost / numNeuronsPre).
    */
   static int calcPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost);

  protected:
   PatchSize() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief If nfp was set to a negative number in params, set it here to the postsynaptic
    * layer's nf
    * @details If nfp is positive in params, it is a fatal error for nfp and post->nf to have
    * different values. It is also a fatal error if there is no ConnectionData component, or
    * more than one, in the CommunicateInitInfo message.
    */
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Called during CommunicateInitInfo stage.
    * If the pre-synaptic layer is a broadcast layer:
    *    If PatchSizeX is negative, set PatchSizeX to the post-synaptic layer's nxGlobal.
    *    If PatchSizeX is post-synaptic layer's nxGlobal, leave it unchanged.
    *    If PatchSizeX is nxGlobal / (# of MPI columns), set it to nxGlobal and send a warning.
    *    If PatchSizeX is anything else, it is a fatal error.
    *
    * If the pre-synaptic layer is not a broadcast layer:
    *    If PatchSizeX is negative or zero, it is a fatal error.
    *    (A positive PatchSizeX value may lead to an error elsewhere in the code, but such a
    *    case would not be flagged here.)
    */
   void setPatchSizeX(HyPerLayer *pre, HyPerLayer *post);

   /**
    * Called during CommunicateInitInfo stage.
    * If the pre-synaptic layer is a broadcast layer:
    *    If PatchSizeY is negative, set PatchSizeY to the post-synaptic layer's nxGlobal.
    *    If PatchSizeY is post-synaptic layer's nxGlobal, leave it unchanged.
    *    If PatchSizeY is nxGlobal / (# of MPI columns), set it to nxGlobal and send a warning.
    *    If PatchSizeY is anything else, it is a fatal error.
    *
    * If the pre-synaptic layer is not a broadcast layer:
    *    If PatchSizeY is negative or zero, it is a fatal error.
    *    (A positive PatchSizeY value may lead to an error elsewhere in the code, but such a
    *    case would not be flagged here.)
    */
   void setPatchSizeY(HyPerLayer *pre, HyPerLayer *post);

   /**
    * Called during CommunicateInitInfo stage. If PatchSizeF is negative, set PatchSizeF to the
    * post-synaptic layer's nf. It is a fatal error if PatchSizeF is >=0 and not equal to the
    * post-synaptic layer's nf.
    */
   void setPatchSizeF(HyPerLayer *pre, HyPerLayer *post);

   void setPatchSizeXorY(
         int &patchSize,
         int correctPatchSize,
         int mpiSize,
         bool isBroadcastPre,
         char const *postName,
         char axis);

  protected:
   int mPatchSizeX = -1;
   int mPatchSizeY = -1;
   int mPatchSizeF = -1;

   ConnectionData *mConnectionData = nullptr;
   bool mWarnDefaultNfp            = true;
   // Whether to print a warning if the default nfp is used.
   // Derived classes can set to false if no warning is necessary.
};

} // namespace PV

#endif // PATCHSIZE_HPP_
