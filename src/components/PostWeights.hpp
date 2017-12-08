/*
 * PostWeights.hpp
 *
 *  Created on: Sep 1, 2017
 *      Author: Pete Schultz
 */

#ifndef POSTWEIGHTS_HPP_
#define POSTWEIGHTS_HPP_

#include "Weights.hpp"

namespace PV {

/**
 * Now that you've read this doxygen comment, you understand the PostWeights class completely.
 * Don't try to tell me you don't.
 */
class PostWeights : public Weights {
  public:
   PostWeights(std::string const &name);

   void initializePostWeights(Weights *preWeights);

   /**
    * Calculates the patch size from the postsynaptic perspective, given the patch size from the
    * presynaptic perspective and the PVLayerLoc structs for the pre- and post-synaptic layers.
    *
    * If numNeuronsPre == numNeuronsPost, the return value is prePatchSize.
    *
    * If numNeuronsPre > numNeuronsPost, numNeuronsPre must be an integer multiple of
    * numNeuronsPost. The return value is prePatchSize * (numNeuronsPre / numNeuronsPost);
    *
    * If numNeuronsPre < numNeuronsPost, numNeuronsPost must be an integer multiple of
    * numNeuronsPre, and prePatchSize must be in integer multiple of their quotient.
    * The return value is the prePatchSize / (numNeuronsPost / numNeuronsPre).
    */
   static int calcPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost);

  private:
   /**
    * Called internally by calcPostPatchSize for the case where numNeuronsPre > numNeuronsPost.
    */
   static int manyToOnePostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost);

   /**
    * Called internally by calcPostPatchSize for the case where numNeuronsPost > numNeuronsPre.
    */
   static int oneToManyPostPatchSize(int prePatchSize, int numNeuronsPre, int numNeuronsPost);
}; // end class PostWeights

} // end namespace PV

#endif // POSTWEIGHTS_HPP_
