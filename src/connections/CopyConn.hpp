/*
 * CopyConn.hpp
 *
 *
 *  Created on: Nov 19, 2014
 *      Author: pschultz
 */

#ifndef COPYCONN_HPP_
#define COPYCONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"

namespace PV {

class CopyConn : public HyPerConn {
  public:
   CopyConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~CopyConn();

  protected:
   CopyConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void createComponentTable(char const *description) override;

   virtual ArborList *createArborList() override;
   virtual PatchSize *createPatchSize() override;
   virtual SharedWeights *createSharedWeights() override;
   virtual WeightsPairInterface *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;
   virtual OriginalConnNameParam *createOriginalConnNameParam();

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class CopyConn

} // namespace PV

#endif // COPYCONN_HPP_
