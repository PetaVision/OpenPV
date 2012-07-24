/*
 * InitGauss2DWeightsParams.hpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#ifndef INITBIDSWEIGHTSPARAMS_HPP_
#define INITBIDSWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"
#include "../layers/BIDSLayer.hpp"

namespace PV {

class InitBIDSWeightsParams: public PV::InitWeightsParams {
public:
   InitBIDSWeightsParams();
   InitBIDSWeightsParams(HyPerConn * parentConn);
   virtual ~InitBIDSWeightsParams();
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
   inline BIDSCoords *getCoords()  {return coords;}
   inline int getNumNodes()        {return numNodes;}

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
   BIDSCoords * coords; //structure array pointer that holds the randomly generated corrdinates for the specified number of BIDS nodes
   int numNodes;
   //calculated values;
   double r2Max;
   double r2Min;
   bool self;

};

} /* namespace PV */
#endif /* INITBIDSWEIGHTSPARAMS_HPP_ */
