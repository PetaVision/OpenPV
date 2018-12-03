/*
 * TransposePatchSize.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef TRANSPOSEPATCHSIZE_HPP_
#define TRANSPOSEPATCHSIZE_HPP_

#include "components/DependentPatchSize.hpp"

namespace PV {

/**
 * A subclass of DependentPatchSize, which computes nxp, nyp, and nfp as the
 * dimensions of a patch of the transpose of a connection specified
 * in an OriginalConnNameParam component.
 */
class TransposePatchSize : public DependentPatchSize {
  public:
   TransposePatchSize(char const *name, PVParams *params, Communicator *comm);
   virtual ~TransposePatchSize();

  protected:
   TransposePatchSize();

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void setPatchSize(PatchSize *originalPatchSize) override;

}; // class TransposePatchSize

} // namespace PV

#endif // TRANSPOSEPATCHSIZE_HPP_
