/*
 * BIDSConn.hpp
 *
 *  Created on: Aug 17, 2012
 *      Author: Brennan Nowers
 */

#ifndef BIDSCONN_HPP_
#define BIDSCONN_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class BIDSConn : public PV::HyPerConn {

public:
   BIDSConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   ~BIDSConn();

protected:
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual void ioParam_lateralRadius(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);

   virtual void ioParam_jitterSource(enum ParamsIOFlag ioFlag);
   virtual void ioParam_jitter(enum ParamsIOFlag ioFlag);
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
