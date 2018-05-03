/*
 * VaryingHyPerConn.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef VARYINGHYPERCONN_HPP_
#define VARYINGHYPERCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class VaryingHyPerConn : public HyPerConn {

  public:
   VaryingHyPerConn(const char *name, HyPerCol *hc);
   virtual ~VaryingHyPerConn();

  protected:
   VaryingHyPerConn() {}
   int initialize(const char *name, HyPerCol *hc);
   BaseWeightUpdater *createWeightUpdater() override;

}; // end class VaryingHyPerConn

} // end namespace PV block

#endif /* VARYINGHYPERCONN_HPP_ */
