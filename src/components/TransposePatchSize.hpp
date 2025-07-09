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
   TransposePatchSize(char const *name, PVParams *params, Communicator const *comm);
   int getOriginalPatchSizeX() const { return mOriginalPatchSizeX; }
   int getOriginalPatchSizeY() const { return mOriginalPatchSizeY; }
   int getOriginalPatchSizeF() const { return mOriginalPatchSizeF; }
   virtual ~TransposePatchSize();

  protected:
   TransposePatchSize();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void setPatchSizeX(HyPerLayer *pre, HyPerLayer *post) override;
   virtual void setPatchSizeY(HyPerLayer *pre, HyPerLayer *post) override;
   virtual void setPatchSizeF(HyPerLayer *pre, HyPerLayer *post) override;

  protected:
   int mOriginalPatchSizeX;
   int mOriginalPatchSizeY;
   int mOriginalPatchSizeF;
}; // class TransposePatchSize

} // namespace PV

#endif // TRANSPOSEPATCHSIZE_HPP_
