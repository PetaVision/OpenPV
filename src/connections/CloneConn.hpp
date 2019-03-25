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

class CloneConn : public HyPerConn {
  public:
   CloneConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~CloneConn();

  protected:
   CloneConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void createComponentTable(char const *description) override;

   virtual BaseDelivery *createDeliveryObject() override;
   virtual ArborList *createArborList() override;
   virtual PatchSize *createPatchSize() override;
   virtual SharedWeights *createSharedWeights() override;
   virtual WeightsPairInterface *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual NormalizeBase *createWeightNormalizer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;
   virtual OriginalConnNameParam *createOriginalConnNameParam();

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class CloneConn

} // namespace PV

#endif // CLONECONN_HPP_
