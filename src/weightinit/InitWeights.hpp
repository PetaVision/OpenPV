/*
 * InitWeights.hpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTS_HPP_
#define INITWEIGHTS_HPP_

#include "columns/BaseObject.hpp"
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

class InitWeights : public BaseObject {
  protected:
   /**
    * List of parameters needed by InitWeights class
    * @name InitWeights Parameters
    * @{
    */

   /**
    * @brief initWeightsFile: A path to a weight pvp file to use for
    * initializing the weights, which overrides the usual method of
    * initializing weights defined by the class being instantiated. If the
    * weights file has fewer features than the connection being initialized,
    * the weights file is used for the lower-indexed features and the
    * calcWeights method is used for the rest. If null or empty, calcWeights()
    * is used to initialize all the weights.
    */
   virtual void ioParam_initWeightsFile(enum ParamsIOFlag ioFlag);

   /**
    * @brief useListOfArborFiles: A flag that indicates whether the
    * initWeightsFile contains a single weight pvp file, or a list of files to
    * be used for several arbors. The files inthe list can themselves have
    * multiple arbors; each file isloaded in turn until the number of arbors
    * in the connection is reached.
    */
   virtual void ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag);

   /**
    * @brief combineWeightFiles: A flag that indicates whether the
    * initWeightsFile is a list of weight files that should be combined.
    */
   virtual void ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag);

   /**
    * @brief numWeightFiles: If combineWeightFiles is set, specifies the
    * number of weight files in the list of files specified by initWeightsFile.
    */
   virtual void ioParam_numWeightFiles(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   InitWeights(char const *name, HyPerCol *hc);
   virtual ~InitWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /*
    * initializeWeights is not virtual.  It checks initFromLastFlag and then
    * filename, loading weights from a file if appropriate.  Otherwise
    * it calls calcWeights with no arguments.
    * For most InitWeights objects, calcWeights(void) does not have to be
    * overridden but calcWeights(dataStart, patchIndex, arborId) should be.
    * For a few InitWeights classes (e.g. InitDistributedWeights),
    * calcWeights(void) is overridden: a fixed number of weights is active,
    * so it is more convenient and efficient to handle all the weights
    * together than to call one patch at a time.
    */
   int initializeWeights(PVPatch ***patches, float **dataStart, double *timef = NULL);

  protected:
   InitWeights();
   int initialize(const char *name, HyPerCol *hc);

   virtual int setDescription();
   virtual int communicateInitInfo(CommunicateInitInfoMessage const *message) override;
   virtual void calcWeights();
   virtual void calcWeights(float *dataStart, int patchIndex, int arborId);

   virtual int readWeights(
         bool sharedWeights,
         float **dataStart,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         const char *filename,
         double *timestampPtr = nullptr);

   virtual int initRNGs(bool isKernel) { return PV_SUCCESS; }
   virtual int zeroWeightsOutsideShrunkenPatch(PVPatch ***patches);
   void readListOfArborFiles(
         bool sharedWeights,
         float **dataStart,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         const char *listOfArborsFilename,
         double *timestampPtr = nullptr);
   void readCombinedWeightFiles(
         bool sharedWeights,
         float **dataStart,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         const char *fileOfWeightFiles,
         double *timestampPtr = nullptr);
   void readWeightPvpFile(
         bool sharedWeights,
         float **dataStart,
         int numPatchesX,
         int numPatchesY,
         int numPatchesF,
         const char *weightPvpFile,
         int numArbors,
         double *timestampPtr = nullptr);

   int kernelIndexCalculations(int patchIndex);
   float calcYDelta(int jPost);
   float calcXDelta(int iPost);
   float calcDelta(int post, float dPost, float distHeadPreUnits);

  private:
   int initialize_base();

  protected:
   char *mFilename           = nullptr;
   bool mUseListOfArborFiles = false;
   bool mCombineWeightFiles  = false;
   int mNumWeightFiles       = 1;
   HyPerConn *mCallingConn   = nullptr;
   HyPerLayer *mPreLayer     = nullptr;
   HyPerLayer *mPostLayer    = nullptr;
   float mDxPost;
   float mDyPost;
   float mXDistHeadPreUnits;
   float mYDistHeadPreUnits;

}; // class InitWeights

} /* namespace PV */
#endif /* INITWEIGHTS_HPP_ */
