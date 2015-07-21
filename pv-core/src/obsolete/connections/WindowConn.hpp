/*
 * WindowConn.hpp
 *
 *  Created on: Nov 25, 2014
 *      Author: pschultz
 */

#ifdef OBSOLETE // Marked obsolete Dec 2, 2014.  Use sharedWeights=false instead of windowing.

#ifndef WINDOWCONN_HPP_
#define WINDOWCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class WindowConn: public HyPerConn {
public:
   WindowConn(char const * name, HyPerCol * hc);
   virtual int defaultUpdateInd_dW(int arbor_ID, int kExt);
   virtual ~WindowConn();

protected:
   WindowConn();
   int initialize(char const * name, HyPerCol * hc);

   virtual void ioParam_useWindowPost(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

protected:
   bool useWindowPost; // shadows a HyPerConn member variable, but HyPerConn's useWindowPost is deprecated.
};

} /* namespace PV */

#endif /* WINDOWCONN_HPP_ */

#endif // OBSOLETE
