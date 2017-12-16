/* MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef MOMENTUMCONN_HPP_
#define MOMENTUMCONN_HPP_

#include "connections/HyPerConn.hpp"

namespace PV {

class HyPerCol;

class MomentumConn : public HyPerConn {
  public:
   MomentumConn(char const *name, HyPerCol *hc);

   virtual ~MomentumConn();

   char const *getMomentumMethod() const;

  protected:
   MomentumConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual BaseWeightUpdater *createWeightUpdater() override;
}; // class MomentumConn

} // namespace PV

#endif // MOMENTUMCONN_HPP_
