/*
 * WeightsPairInterface.hpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#ifndef WEIGHTSPAIRINTERFACE_HPP_
#define WEIGHTSPAIRINTERFACE_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "components/PatchSize.hpp"
#include "components/Weights.hpp"

namespace PV {

class WeightsPairInterface : public BaseObject {
  public:
   WeightsPairInterface(char const *name, HyPerCol *hc);

   virtual ~WeightsPairInterface();

   /**
    * Objects that need the presynaptic weights should call this function,
    * but should make sure that the WeightsPairInterface object has completed
    * its communicate stage first.
    *
    * Internally, this method calls the virtual createPreWeights() method.
    */
   void needPre();

   /**
    * Objects that need the postsynaptic weights should call this function,
    * but should make sure that the WeightsPairInterface object has completed
    * its communicate stage first.
    *
    * Internally, this method calls the virtual createPostWeights() method.
    */
   void needPost();

   Weights *getPreWeights() { return mPreWeights; }
   Weights *getPostWeights() { return mPostWeights; }

   ConnectionData const *getConnectionData() { return mConnectionData; }

  protected:
   WeightsPairInterface() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * A pure virtual method for creating the presynaptic weights, called by
    * needPre() if mPreWeights is null and the communicate stage has been completed.
    *
    * Implementations can assume that mPreWeights is null on entry, and should set that
    * member variable.
    */
   virtual void createPreWeights(std::string const &weightsName) = 0;

   /**
    * A pure virtual method for creating the postsynaptic weights, called by
    * needPost() if mPostWeights is null and the communicate stage has been completed.
    *
    * Implementations can assume that mPostWeights is null on entry, and should set that
    * member variable.
    */
   virtual void createPostWeights(std::string const &weightsName) = 0;

   virtual Response::Status allocateDataStructures() override;

   virtual void allocatePreWeights();

   virtual void allocatePostWeights();

  protected:
   PatchSize *mPatchSize           = nullptr;
   ConnectionData *mConnectionData = nullptr;

   Weights *mPreWeights  = nullptr;
   Weights *mPostWeights = nullptr;
};

} // namespace PV

#endif // WEIGHTSPAIRINTERFACE_HPP_
