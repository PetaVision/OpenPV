/*
 * SharedConnDebugInitWeights.hpp
 *
 *  Created on: Aug 22, 2011
 *      Author: kpeterson
 */

#ifndef SHAREDCONNDEBUGINITWEIGHTS_HPP_
#define SHAREDCONNDEBUGINITWEIGHTS_HPP_

#include <connections/HyPerConn.hpp>

namespace PV {

class SharedConnDebugInitWeights : public PV::HyPerConn {
  protected:
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);

  public:
   SharedConnDebugInitWeights();
   SharedConnDebugInitWeights(const char *name, HyPerCol *hc);
   virtual ~SharedConnDebugInitWeights();

  protected:
   int initialize(const char *name, HyPerCol *hc);

   virtual SharedWeights *createSharedWeights() override;

   virtual InitWeights *createWeightInitializer() override { return nullptr; }
   // This class computes weights without using InitWeights class,
   // in order to compare to connections that do use the weightInitializer.

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status initializeState() override;

   void initializeGaussian2DWeights(float *dataStart, int numPatches);
   void gauss2DCalcWeights(
         float *dataStart,
         int kPre,
         int noPost,
         int numFlanks,
         float shift,
         float rotate,
         float aspect,
         float sigma,
         float r2Max,
         float r2Min,
         float strength,
         float deltaThetaMax,
         float thetaMax,
         float bowtieFlag,
         float bowtieAngle);
   void initializeCocircWeights(float *dataStart, int numPatches);
   void cocircCalcWeights(
         float *dataStart,
         int kPre,
         int noPre,
         int noPost,
         float sigma_cocirc,
         float sigma_kurve,
         float sigma_chord,
         float delta_theta_max,
         float cocirc_self,
         float delta_radius_curvature,
         int numFlanks,
         float shift,
         float aspect,
         float rotate,
         float sigma,
         float r2Max,
         float strength);
   void initializeSmartWeights(float *dataStart, int numPatches);
   void smartWeights(float *dataStart, int k);
   void gaborWeights(
         float *dataStart,
         int xScale,
         int yScale,
         float aspect,
         float sigma,
         float r2Max,
         float lambda,
         float strength,
         float phi);
   void initializeGaborWeights(float *dataStart, int numPatches);

   int
   dataIndexToUnitCellIndex(int dataIndex, int *kx = nullptr, int *ky = nullptr, int *kf = nullptr);

  protected:
   char *mWeightInitTypeString = nullptr;
};

} /* namespace PV */
#endif /* SHAREDCONNDEBUGINITWEIGHTS_HPP_ */
