/*
 * AvConn.hpp
 *
 *  Created on: Oct 9, 2009
 *      Author: rasmussn
 */

#ifndef AVGCONN_HPP_
#define AVGCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class AvgConn: public PV::HyPerConn {
public:

   AvgConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
          int channel, HyPerConn * delagate);
   virtual ~AvgConn();

   virtual int deliver(Publisher * pub, PVLayerCube * cube, int neighbor);
   virtual int write(const char * filename);

protected:

   int initialize(HyPerConn * companion);

   PVLayerCube * avgActivity;
   HyPerConn   * delegate;
};

} // namespace PV

#endif /* AVGCONN_HPP_ */
