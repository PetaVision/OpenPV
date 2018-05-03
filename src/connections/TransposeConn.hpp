/* TransposeConn.cpp
 *
 * Created on: May 16, 2011
 *     Author: peteschultz
 */

#ifndef TRANSPOSECONN_HPP_
#define TRANSPOSECONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"

namespace PV {

class TransposeConn : public HyPerConn {
  public:
   TransposeConn(char const *name, HyPerCol *hc);

   virtual ~TransposeConn();

  protected:
   TransposeConn();

   virtual void defineComponents() override;

   int initialize(char const *name, HyPerCol *hc);

   virtual ArborList *createArborList() override;
   virtual PatchSize *createPatchSize() override;
   virtual SharedWeights *createSharedWeights() override;
   virtual WeightsPairInterface *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual NormalizeBase *createWeightNormalizer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;
   virtual OriginalConnNameParam *createOriginalConnNameParam();

   virtual Response::Status initializeState() override;

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class TransposeConn

} // namespace PV

#endif // TRANSPOSECONN_HPP_
