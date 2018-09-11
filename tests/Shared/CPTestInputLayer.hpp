/*
 * CPTestInputLayer.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef CPTESTINPUTLAYER_HPP_
#define CPTESTINPUTLAYER_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV {

class CPTestInputLayer : public HyPerLayer {

  public:
   CPTestInputLayer(const char *name, HyPerCol *hc);
   virtual ~CPTestInputLayer();
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   virtual Response::Status updateState(double timed, double dt) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   void initializeV();

}; // end class CPTestInputLayer

} // end of namespace PV block

#endif /* CPTESTINPUTLAYER_HPP_ */
