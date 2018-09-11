/*
 * InitializeFromCheckpointFlag.hpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#ifndef INITIALIZEFROMCHECKPOINTFLAG_HPP_
#define INITIALIZEFROMCHECKPOINTFLAG_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the phase parameter from the params file.
 */
class InitializeFromCheckpointFlag : public BaseObject {
  protected:
   /**
    * List of parameters needed from the InitializeFromCheckpointFlag class
    * @name InitializeFromCheckpointFlag Parameters
    * @{
    */

   /**
    * @brief initializeFromCheckpointFlag: specifies the value of the initializeFromCheckpointFlag
    * parameter. Layers use initializeFromCheckpointFlag to determine whether to load from
    * checkpoint during initialization.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /** @} */ // end of InitializeFromCheckpointFlag parameters

  public:
   InitializeFromCheckpointFlag(char const *name, HyPerCol *hc);

   virtual ~InitializeFromCheckpointFlag();

   int getInitializeFromCheckpointFlag() const { return mInitializeFromCheckpointFlag; }

  protected:
   InitializeFromCheckpointFlag() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   bool mInitializeFromCheckpointFlag = false;
};

} // namespace PV

#endif // INITIALIZEFROMCHECKPOINTFLAG_HPP_
