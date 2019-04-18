/*
 * PlasticTestConn.hpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#ifndef PLASTICTESTCONN_HPP_
#define PLASTICTESTCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class PlasticTestConn : public HyPerConn {
  public:
   PlasticTestConn(const char *name, HyPerCol *hc);
   virtual ~PlasticTestConn();

  protected:
   BaseWeightUpdater *createWeightUpdater() override;
}; // end class PlasticTestConn

} // end namespace PV
#endif /* PLASTICTESTCONN_HPP_ */
