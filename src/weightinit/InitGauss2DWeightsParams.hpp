/*
 * InitGauss2DWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITGAUSS2DWEIGHTSPARAMS_HPP_
#define INITGAUSS2DWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitGauss2DWeightsParams: public PV::InitWeightsParams {
public:
   InitGauss2DWeightsParams();
   InitGauss2DWeightsParams(HyPerConn * parentConn);
   virtual ~InitGauss2DWeightsParams();
   void calcOtherParams(int patchIndex);
   bool isSameLocOrSelf(float xDelta, float yDelta, int fPost);
   bool checkBowtieAngle(float xp, float yp);

   //get-set methods:
   inline float getaspect()        {return aspect;}
   inline float getshift()        {return shift;}
   inline int getnumFlanks()        {return numFlanks;}
   inline float getsigma()        {return sigma;}
   inline double getr2Max()        {return r2Max;}
   inline double getr2Min()        {return r2Min;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);


private:

   //params file values:
   float aspect; // circular (not line oriented)
   float sigma;
   float rMax;
   float rMin;  // minimum radius for any connection
   float strength;
   int numFlanks;
   float shift;
   bool bowtieFlag;  // flag for setting bowtie angle
   float bowtieAngle;  // bowtie angle

   //calculated values;
   double r2Max;
   double r2Min;
   bool self;

};

} /* namespace PV */
#endif /* INITGAUSS2DWEIGHTSPARAMS_HPP_ */
