/*
 * InitWindowed3DGaussWeightsParams.hpp
 *
 *  Created on: Jan 18, 2012
 *      Author: kpeterson
 */

#ifndef INITWINDOWED3DGAUSSWEIGHTSPARAMS_HPP_
#define INITWINDOWED3DGAUSSWEIGHTSPARAMS_HPP_

#include "Init3DGaussWeightsParams.hpp"

namespace PV {

class InitWindowed3DGaussWeightsParams: public PV::Init3DGaussWeightsParams {
public:
   InitWindowed3DGaussWeightsParams();
   InitWindowed3DGaussWeightsParams(HyPerConn * parentConn);
   virtual ~InitWindowed3DGaussWeightsParams();

   inline float getWindowShift()        {return windowShift;}
   inline float getWindowShiftT()        {return windowShiftT;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:
   float windowShift;
   float windowShiftT;

};

} /* namespace PV */
#endif /* INITWINDOWED3DGAUSSWEIGHTSPARAMS_HPP_ */
