/*
 * BoundaryConditions.hpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#ifndef BOUNDARYCONDITIONS_HPP_
#define BOUNDARYCONDITIONS_HPP_

#include "columns/BaseObject.hpp"

namespace PV {

/**
 * A component to contain the numAxonalArbors parameter and the delay parameters array.
 */
class BoundaryConditions : public BaseObject {
  protected:
   /**
    * List of parameters needed from the BoundaryConditions class
    * @name BoundaryConditions Parameters
    * @{
    */

   /**
    * @brief mirrorBCflag: If set to true, the margin will mirror the data
    */
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag);

   /**
    * @brief valueBC: If mirrorBC is set to true, Uses the specified value for the margin area
    */
   virtual void ioParam_valueBC(enum ParamsIOFlag ioFlag);

   /** @} */ // end of BoundaryConditions parameters

  public:
   BoundaryConditions(char const *name, HyPerCol *hc);
   virtual ~BoundaryConditions();

   virtual void setObjectType() override;

   /**
    * Use the BoundaryConditions parameters to fill in the extended part of the given
    * buffer using the given PVLayerLoc. The boundary conditions are applied to all
    * elements in the batch.
    */
   virtual void applyBoundaryConditions(float *buffer, PVLayerLoc const *loc) const;

   bool getMirrorBCflag() const { return mMirrorBCflag; }
   float getValueBC() const { return mValueBC; }

  protected:
   BoundaryConditions();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void
   mirrorInteriorToBorder(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToNorthWest(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToNorth(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToNorthEast(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToWest(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToEast(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToSouthWest(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToSouth(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;
   virtual void
   mirrorToSouthEast(float const *srcBuffer, float *destBuffer, PVLayerLoc const *loc) const;

   virtual void fillWithValue(float *buffer, PVLayerLoc const *loc) const;

  protected:
   bool mMirrorBCflag = false;
   float mValueBC     = 0.0f;

}; // class BoundaryConditions

} // namespace PV

#endif // BOUNDARYCONDITIONS_HPP_
