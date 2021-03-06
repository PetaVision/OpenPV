/*
 * IndexWeightConn.hpp
 *
 *  Created on: Mar 2, 2017
 *      Author: pschultz
 *
 * A connection class with a built-in initialization method and a simple
 * update rule, to be used in testing.
 * The weights in each nxp-by-nyp-by-nfp are ordered 0,1,2,...,(nxp*nyp*nfp-1)
 * in standard PetaVision ordering. At initialization, the strength of the
 * weight at index k is k + start-time; when LayerUpdateState is called, the
 * weight at index k becomes k + simulation-time.
 */

#ifndef INDEXWEIGHTCONN_HPP_
#define INDEXWEIGHTCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class IndexWeightConn : public HyPerConn {
  public:
   IndexWeightConn(const char *name, PVParams *params, Communicator const *comm);
   virtual ~IndexWeightConn();

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual InitWeights *createWeightInitializer() override;
   virtual BaseWeightUpdater *createWeightUpdater() override;
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

}; // end class IndexWeightConn

} // end namespace PV block

#endif /* INDEXWEIGHTCONN_HPP_ */
