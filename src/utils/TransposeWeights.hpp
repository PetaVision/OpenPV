/*
 * TransposeWeights.hpp
 *
 *  Created on: Sep 1, 2017
 *      Author: peteschultz
 */

#ifndef TRANSPOSEWEIGHTS_HPP_
#define TRANSPOSEWEIGHTS_HPP_

#include "columns/Communicator.hpp"
#include "components/Weights.hpp"

namespace PV {

class TransposeWeights {
  public:
   static void transpose(Weights *preWeights, Weights *postWeights, Communicator *comm);
   static void transpose(Weights *preWeights, Weights *postWeights, Communicator *comm, int arbor);

  private:
   static void transposeShared(Weights *preWeights, Weights *postWeights, int arbor);
   static void
   transposeNonshared(Weights *preWeights, Weights *postWeights, Communicator *comm, int arbor);
};

} // namespace PV

#endif /* TRANSPOSEWEIGHTS_HPP_ */
