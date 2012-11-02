/*
 * LCALIFLateralProbe.hpp
 *
 *  Created on: Oct 30, 2012
 *      Author: slundquist
 */

#ifndef LCALIFLATERALPROBE_HPP_
#define LCALIFLATERALPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/LCALIFLateralConn.hpp"
#include <assert.h>


namespace PV {

class LCALIFLateralProbe: public BaseConnectionProbe {
   //Methods
public:
   LCALIFLateralProbe();
   LCALIFLateralProbe(const char * probename, const char * filename, HyPerConn * conn, int preIndex);
   LCALIFLateralProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPre, int kyPre, int kfPre);
   virtual ~LCALIFLateralProbe();

   virtual int outputState(double timef);

protected:
   int initialize(const char * probename, const char * filename, HyPerConn * conn, PatchIDMethod method, int preIndex, int kxPre, int kyPre, int kfPre, bool isPostProbe);

private:
   int initialize_base();
   LCALIFLateralConn * LCALIFConn;
   int kLocalRes;
   int kLocalExt;
   int inBounds;
   pvdata_t* postWeights;

   //output variables
   float postIntTr;
   float * preWeights;

};
} // end namespace PV



#endif /* LCALIFLATERALPROBE_HPP_ */
