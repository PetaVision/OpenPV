/*
 * CopyWeightsPair.hpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#ifndef COPYWEIGHTSPAIR_HPP_
#define COPYWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

/**
 * A derived class of WeightsPair that copies weights from another connection, specified
 * in the originalConnName parameter. The nxp, nyp, nfp, and sharedWeights parameters
 * are copied from the original connection, not read from the params file.
 * The weights are copied from the original using the copy() method (which is called
 * by CopyConn::initializeState and CopyUpdater::updateState).
 *
 */
class CopyWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the CopyWeightsPair class
    * @name CopyWeightsPair Parameters
    * @{
    */

   /**
    * @brief nxp: CopyWeightsPair does not read the nxp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nyp: CopyWeightsPair does not read the nyp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nfp: CopyWeightsPair does not read the nfp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: CopyWeightsPair does not read the sharedWeights parameter,
    * but inherits it from the originalConn's WeightsPair.
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of CopyWeightsPair parameters

  public:
   CopyWeightsPair(char const *name, HyPerCol *hc);

   virtual ~CopyWeightsPair();

   virtual void needPre();
   virtual void needPost();

   void copy();

   char const *getOriginalConnName() const { return mOriginalConnName; }

   WeightsPair const *getOriginalWeightsPair() const { return mOriginalWeightsPair; }

  protected:
   CopyWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void copyParameters();

  protected:
   char *mOriginalConnName = nullptr;

   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // COPYWEIGHTSPAIR_HPP_
