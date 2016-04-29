/*
 * HyPerConnDebugInitWeights.hpp
 *
 *  Created on: Aug 16, 2011
 *      Author: kpeterson
 */

#ifndef HYPERCONNDEBUGINITWEIGHTS_HPP_
#define HYPERCONNDEBUGINITWEIGHTS_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class HyPerConnDebugInitWeights: public PV::HyPerConn {
public:
   HyPerConnDebugInitWeights();
   HyPerConnDebugInitWeights(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~HyPerConnDebugInitWeights();

   virtual int communicateInitInfo();
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
         const char * filename);


protected:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);
   virtual void ioParam_copiedConn(enum ParamsIOFlag ioFlag);
   virtual void readChannelCode(PVParams * params) { channel = CHANNEL_INH;}

   PVPatch ** initializeGaussian2DWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches);
   virtual int gauss2DCalcWeights(PVPatch * wp, pvdata_t * dataStart, int kPre, int noPost,
                             int numFlanks, float shift, float rotate, float aspect, float sigma,
                             float r2Max, float r2Min, float strength, float deltaThetaMax, float thetaMax,
                             float bowtieFlag, float bowtieAngle);
   PVPatch ** initializeCocircWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches);
   virtual int cocircCalcWeights(PVPatch * wp, pvdata_t * dataStart, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
         float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
         float aspect, float rotate, float sigma, float r2Max, float strength);
   PVPatch ** initializeSmartWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches);
   int smartWeights(PVPatch * wp, pvdata_t * dataStart, int k);
   int gaborWeights(PVPatch * wp, pvdata_t * dataStart, int xScale, int yScale,
                    float aspect, float sigma, float r2Max, float lambda, float strength, float phi);
   PVPatch ** initializeGaborWeights(PVPatch ** patches, pvdata_t * dataStart, int numPatches);
   int copyToWeightPatch(PVPatch * sourcepatch, int arbor, int patchindex);

private:
   int initialize_base();
   char * otherConnName;
   HyPerConn *otherConn;
};

BaseObject * createHyPerConnDebugInitWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* HYPERCONNDEBUGINITWEIGHTS_HPP_ */
