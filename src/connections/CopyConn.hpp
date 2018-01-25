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

class HyPerCol;

class CopyConn : public HyPerConn {
  public:
   CopyConn(char const *name, HyPerCol *hc);

   virtual ~CopyConn();

  protected:
   CopyConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual ArborList *createArborList() override;
   virtual PatchSize *createPatchSize() override;
   virtual SharedWeights *createSharedWeights() override;
   virtual WeightsPairInterface *createWeightsPair() override;
   virtual InitWeights *createWeightInitializer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;
   virtual OriginalConnNameParam *createOriginalConnNameParam();

   virtual Response::Status initializeState() override;

  protected:
   OriginalConnNameParam *mOriginalConnNameParam = nullptr;
}; // class CopyConn

} // namespace PV

#endif // COPYCONN_HPP_
