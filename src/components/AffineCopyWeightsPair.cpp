/*
 * AffineCopyWeightsPair.cpp
 *
 *  Created on: June 27, 2019
 *      Author: Xinhua Zhang
 */

#include <vector>
#include "AffineCopyWeightsPair.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "components/Weights.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {
   AffineCopyWeightsPair::AffineCopyWeightsPair(char const *name, PVParams *params, Communicator const *comm) {
      initialize(name, params, comm);
   }
   
   AffineCopyWeightsPair::~AffineCopyWeightsPair() {}

   void AffineCopyWeightsPair::initialize(char const *name, PVParams *params, Communicator const *comm) {
      WeightsPair::initialize(name, params, comm);
   }

   void AffineCopyWeightsPair::ioParam_angleOfRotation(enum ParamsIOFlag ioFlag)
   {
      parameters()->ioParamValue(ioFlag, name, "angleOfRotation", &mAngle, mAngle, false);
   }

   int AffineCopyWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) 
   {
      WeightsPair::ioParamsFillGroup(ioFlag);      
      ioParam_angleOfRotation(ioFlag);
      return PV_SUCCESS;      
   }

   Response::Status AffineCopyWeightsPair::allocateDataStructures() 
   {
      CopyWeightsPair::allocateDataStructures();
      
      Weights *originalPreWeights = mOriginalWeightsPair->getPreWeights();
      int const nxp = originalPreWeights->getPatchSizeX();
      int const nyp = originalPreWeights->getPatchSizeY();
      int const nfp = originalPreWeights->getPatchSizeF();
      

      pvAssert(nxp == getPreWeights()->getPatchSizeX());
      pvAssert(nyp == getPreWeights()->getPatchSizeY());
      pvAssert(nfp == getPreWeights()->getPatchSizeF());
      

#ifdef PV_USE_OPENMP_THREADS
      int const nf = originalPreWeights->getNumDataPatches();
      pvAssert(nf == getPreWeights()->getNumDataPatches());
      
      inputBuffer = vector<vector<Mat>>(nf,vector<Mat>(nfp));
      outputBuffer = vector<vector<Mat>>(nf,vector<Mat>(nfp));

      // Mat in OpenCV is shared if it is not initialized
      for (int i = 0; i < nf; i++) {
         for (int j = 0; j < nfp; j++) {
            inputBuffer[i][j].create(nxp,nyp,DataType<float>::type);
            outputBuffer[i][j].create(nxp,nyp,DataType<float>::type);
         }
      }
      
      rot_mat = getRotationMatrix2D(Point(outputBuffer[0][0].size()/ 2) , mAngle, 1.0);
            
      inputPatchBuffer = vector<float *>(nf, NULL);
      outputPatchBuffer = vector<float *>(nf, NULL);
#else
      
      inputBuffer = vector<Mat>(nfp);
      outputBuffer = vector<Mat>(nfp);
      // Mat in OpenCV is shared if it is not initialized
      for (int i = 0; i < nfp; i++) {
            inputBuffer[i].create(nxp,nyp,DataType<float>::type);
            outputBuffer[i].create(nxp,nyp,DataType<float>::type);
      }
      
      rot_mat = getRotationMatrix2D(Point(outputBuffer[0].size()/ 2) , mAngle, 1.0);
            
      inputPatchBuffer = NULL;
      outputPatchBuffer = NULL;   
#endif
      return Response::SUCCESS;
   }
   

   void AffineCopyWeightsPair::setObjectType() { mObjectType = "AffineCopyWeightsPair"; }

   void AffineCopyWeightsPair::copy() {
      
      Weights *originalPreWeights = mOriginalWeightsPair->getPreWeights();
      int const nxp = originalPreWeights->getPatchSizeX();
      int const nyp = originalPreWeights->getPatchSizeY();
      int const nfp = originalPreWeights->getPatchSizeF();
      int const nf = originalPreWeights->getNumDataPatches();
      int const numArbors = originalPreWeights->getNumArbors(); 


      for (int arbor = 0; arbor < numArbors; arbor++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
         for (int i = 0; i < nf; i++) {
            inputPatchBuffer[i] = originalPreWeights->getDataFromDataIndex(arbor, i);
            outputPatchBuffer[i] = getPreWeights()->getDataFromDataIndex(arbor, i);

            // extract a weight patch
            for (int iy = 0 ; iy < nyp; iy++) 
               for (int ix = 0 ; ix < nxp; ix++)   
                  for (int k = 0; k < nfp; k++)
                     inputBuffer[i][k].at<float>(ix,iy) = inputPatchBuffer[i][iy*(nxp*nfp) + ix * nfp + k];
            
            // rotate
            for (int k = 0; k < nfp; k++) 
               warpAffine(inputBuffer[i][k], outputBuffer[i][k], rot_mat, inputBuffer[i][k].size());            

            // store the rotated weight patch in its own pre weight
            for (int iy = 0 ; iy < nyp; iy++) 
               for (int ix = 0 ; ix < nxp; ix++) 
                  for (int k = 0; k < nfp; k++)
                     outputPatchBuffer[i][iy*(nxp*nfp) + ix * nfp + k] = outputBuffer[i][k].at<float>(ix,iy);            
#else
            for (int i = 0; i < nf; i++) {
               inputPatchBuffer = originalPreWeights->getDataFromDataIndex(arbor, i);
               outputPatchBuffer = getPreWeights()->getDataFromDataIndex(arbor, i);

               // extract a weight patch
               for (int iy = 0 ; iy < nyp; iy++) 
                  for (int ix = 0 ; ix < nxp; ix++)   
                     for (int k = 0; k < nfp; k++)
                        inputBuffer[k].at<float>(ix,iy) = inputPatchBuffer[iy*(nxp*nfp) + ix * nfp + k];
            
               // rotate
               for (int k = 0; k < nfp; k++) 
                  warpAffine(inputBuffer[k], outputBuffer[k], rot_mat, inputBuffer[k].size());            

               // store the rotated weight patch in its own pre weight
               for (int iy = 0 ; iy < nyp; iy++) 
                  for (int ix = 0 ; ix < nxp; ix++) 
                     for (int k = 0; k < nfp; k++)
                        outputPatchBuffer[iy*(nxp*nfp) + ix * nfp + k] = outputBuffer[k].at<float>(ix,iy);            
#endif
         }         
      }
   }   
} // namespace PV

