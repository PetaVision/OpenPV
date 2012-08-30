/*
 * InitBIDSLateralParams.hpp
 *
 *  Created on: Aug 10, 2012
 *      Author: bnowers
 */

#ifndef INITBIDSLATERALPARAMS_HPP_
#define INITBIDSLATERALPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "../layers/BIDSLayer.hpp"
#include "../layers/BIDSMovieCloneMap.hpp"

namespace PV {

class InitBIDSLateralParams: public PV::InitWeightsParams {
public:
   InitBIDSLateralParams();
   InitBIDSLateralParams(HyPerConn * parentConn);
   virtual ~InitBIDSLateralParams();
   void calcOtherParams(int patchIndex);
   bool isSameLocOrSelf(float xDelta, float yDelta, int fPost);
   bool checkBowtieAngle(float xp, float yp);

   //get-set methods:
   inline float getaspect()        {return aspect;}
   inline float getshift()        {return shift;}
   inline int getnxp()             {return nxp;}
   inline int getnyp()             {return nyp;}
   inline int getnumFlanks()        {return numFlanks;}
   inline float getsigma()        {return sigma;}
   inline double getr2Max()        {return r2Max;}
   inline double getr2Min()        {return r2Min;}
   inline BIDSCoords *getCoords()  {return coords;}
   inline const char * getFalloffType()  {return falloffType;}
   inline int getLateralRadius()   {return lateralRadius;}
   inline HyPerConn * getParentConn() {return parentConn;}
   inline float getStrength() {return strength;}

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
   int nxp;
   int nyp;
   int numFlanks;
   float shift;
   bool bowtieFlag;  // flag for setting bowtie angle
   float bowtieAngle;  // bowtie angle
   BIDSCoords * coords; //structure array pointer that holds the randomly generated corrdinates for the specified number of BIDS nodes
   const char * falloffType;
   int lateralRadius;
   //calculated values;
   double r2Max;
   double r2Min;
   bool self;
};

} /* namespace PV */
#endif /* INITBIDSLATERALPARAMS_HPP_ */
