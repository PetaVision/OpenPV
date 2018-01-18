/*
 * ImpliedWeights.hpp
 *
 *  Created on: Jul 28, 2017
 *      Author: Pete Schultz
 */

#ifndef IMPLIEDWEIGHTS_HPP_
#define IMPLIEDWEIGHTS_HPP_

#include "components/Weights.hpp"

namespace PV {

/**
 * ImpliedWeights is derived from the Weights class.
 * It constructs the PatchGeometry the same way as the base class,
 * but the number of data patches is zero in all dimensions, and
 * the Data vector is empty.
 *
 * ImpliedWeights was motivated by PoolingConn, which uses
 * HyPerConn-style patch geometry, but the weights are all equal.
 *
 * Since the Data vector is always empty, the
 * getData, getDataFromPatchIndex, and getDataFromDataIndex methods
 * always throw out_of_range exceptions. The calcMinWeight methods
 * return a large-magnitude positive number, and the calcMaxWeight
 * methods return a large-magnitude negative number.
 */
class ImpliedWeights : public Weights {

  public:
   ImpliedWeights(std::string const &name);

   ImpliedWeights(
         std::string const &name,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         PVLayerLoc const *preLoc,
         PVLayerLoc const *postLoc,
         double timestamp);

   virtual ~ImpliedWeights() {}

  private:
   virtual void initNumDataPatches() override;
}; // end class ImpliedWeights

} // end namespace PV

#endif // IMPLIEDWEIGHTS_HPP_
