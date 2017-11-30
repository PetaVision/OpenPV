/*
 * HyPerConn.hpp
 *
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "components/ConnectionData.hpp"
#include "components/WeightsPair.hpp"
#include "connections/BaseConnection.hpp"
//#include "components/WeightUpdater.hpp"
//#include "normalizers/NormalizeBase.hpp"
#include "weightinit/InitWeights.hpp"

namespace PV {

class HyPerCol;

class HyPerConn : public BaseConnection {
  public:
   HyPerConn(char const *name, HyPerCol *hc);

   virtual ~HyPerConn();

  protected:
   HyPerConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void defineComponents() override;

   virtual WeightsPair *createWeightsPair();
   virtual InitWeights *createWeightInitializer();
   // virtual NormalizeBase *createWeightNormalizer();
   virtual BaseDelivery *createDeliveryObject() override;
   // virtual WeightUpdater *createWeightUpdater() override;

   virtual int initializeState() override;

  protected:
   char *mNormalizeMethod = nullptr;

}; // class HyPerConn

} // namespace PV

#endif // HYPERCONN_HPP_
