/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#ifndef CLONECONN_HPP_
#define CLONECONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"

namespace PV {

class HyPerCol;

class CloneConn : public HyPerConn {
  public:
   CloneConn(char const *name, HyPerCol *hc);

   virtual ~CloneConn();

  protected:
   CloneConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual BaseDelivery *createDeliveryObject() override;
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
}; // class CloneConn

} // namespace PV

#endif // CLONECONN_HPP_
