/*
 * BIDSConn.hpp
 *
 *  Created on: Aug 17, 2012
 *      Author: Brennan Nowers
 */

#ifndef BIDSCONN_HPP_
#define BIDSCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class BIDSConn : public PV::HyPerConn {

public:
   BIDSConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name,
             const char * filename, InitWeights *weightInit);
   ~BIDSConn();

protected:
   virtual int setParams(PVParams * params);
   virtual void readLateralRadius(PVParams * inputParams);
   virtual void readJitterSource(PVParams * inputParams);
   virtual void readJitter(PVParams * inputParams);
   virtual int setPatchSize();

private:
   int initialize_base();

// Member variables
protected:
   double lateralRadius;
   char * jitterSourceName;
   double jitter;
};

} // namespace PV

#endif /* BIDSCONN_HPP_ */
