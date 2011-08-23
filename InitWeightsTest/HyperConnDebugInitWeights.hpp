/*
 * HyperConnDebugInitWeights.hpp
 *
 *  Created on: Aug 16, 2011
 *      Author: kpeterson
 */

#ifndef HYPERCONNDEBUGINITWEIGHTS_HPP_
#define HYPERCONNDEBUGINITWEIGHTS_HPP_

#include "../PetaVision/src/connections/HyPerConn.hpp"

namespace PV {

class HyperConnDebugInitWeights: public PV::HyPerConn {
public:
   HyperConnDebugInitWeights();
   HyperConnDebugInitWeights(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, HyPerConn *copiedConn);
   virtual ~HyperConnDebugInitWeights();

   virtual int initialize_base();
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, HyPerConn *copiedConn);
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
         const char * filename);


protected:
   PVPatch ** initializeGaussian2DWeights(PVPatch ** patches, int numPatches);
   virtual int gauss2DCalcWeights(PVPatch * wp, int kPre, int noPost,
                             int numFlanks, float shift, float rotate, float aspect, float sigma,
                             float r2Max, float strength, float deltaThetaMax, float thetaMax,
                             float bowtieFlag, float bowtieAngle);
   PVPatch ** initializeCocircWeights(PVPatch ** patches, int numPatches);
   virtual int cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
         float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
         float aspect, float rotate, float sigma, float r2Max, float strength);
   PVPatch ** initializeSmartWeights(PVPatch ** patches, int numPatches);
   int smartWeights(PVPatch * wp, int k);
   int gaborWeights(PVPatch * wp, int xScale, int yScale,
                    float aspect, float sigma, float r2Max, float lambda, float strength, float phi);
   PVPatch ** initializeGaborWeights(PVPatch ** patches, int numPatches);


private:
   HyPerConn *otherConn;
};

} /* namespace PV */
#endif /* HYPERCONNDEBUGINITWEIGHTS_HPP_ */
