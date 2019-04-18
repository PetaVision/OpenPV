/*
 * DependentPatchSize.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef DEPENDENTPATCHSIZE_HPP_
#define DEPENDENTPATCHSIZE_HPP_

#include "components/PatchSize.hpp"

namespace PV {

/**
 * A subclass of PatchSize, which retrieves nxp, nyp, and nfp from the connection named
 * in an OriginalConnNameParam component, instead of reading them from params.
 */
class DependentPatchSize : public PatchSize {
  protected:
   /**
    * List of parameters needed from the DependentPatchSize class
    * @name DependentPatchSize Parameters
    * @{
    */

   /**
    * @brief nxp: DependentPatchSize does not read the nxp parameter,
    * but copies it from the original connection.
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nyp: DependentPatchSize does not read the nyp parameter,
    * but copies it from the original connection.
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nfp: DependentPatchSize does not read the nfp parameter,
    * but copies it from the original connection.
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of DependentPatchSize parameters

  public:
   DependentPatchSize(char const *name, HyPerCol *hc);
   virtual ~DependentPatchSize();

   virtual void setObjectType() override;

  protected:
   DependentPatchSize();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   char const *getOriginalConnName(std::map<std::string, Observer *> const hierarchy) const;
   PatchSize *getOriginalPatchSize(
         std::map<std::string, Observer *> const hierarchy,
         char const *originalConnName) const;

   virtual void setPatchSize(PatchSize *originalPatchSize);

}; // class DependentPatchSize

} // namespace PV

#endif // DEPENDENTPATCHSIZE_HPP_
