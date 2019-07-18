/*
 * AffineCopyWeightsPair.hpp
 *
 *  Created on: June 27, 2019
 *      Author: Xinhua Zhang
 */
#ifndef AFFINECOPYWEIGHTSPAIR_HPP_
#define AFFINECOPYWEIGHTSPAIR_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "components/CopyWeightsPair.hpp"
#include "opencv2/opencv.hpp"

#define HAVE_OPENCV_IMGPROC
using namespace cv;
using namespace std;

namespace PV{
   
class AffineCopyWeightsPair : public CopyWeightsPair {
   
public:
   AffineCopyWeightsPair(char const *name, PVParams *params, Communicator const *comm);
   virtual ~AffineCopyWeightsPair();

   virtual void copy() override;

protected:
   AffineCopyWeightsPair() {}
   
   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;
   
   virtual void ioParam_angleOfRotation(enum ParamsIOFlag ioFlag);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status allocateDataStructures() override;

private:
   float mAngle = 0;
   Mat rot_mat;
#ifdef PV_USE_OPENMP_THREADS
   vector<vector<Mat>> inputBuffer,outputBuffer;
   vector<float *> inputPatchBuffer,outputPatchBuffer;
#else
   vector<Mat> inputBuffer,outputBuffer;
   float *inputPatchBuffer,*outputPatchBuffer;
#endif
   
};
   
} // namespace PV
#endif // AFFINECOPYWEIGHTSPAIR_HPP_
