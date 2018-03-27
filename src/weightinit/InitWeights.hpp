/*
 * InitWeights.hpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTS_HPP_
#define INITWEIGHTS_HPP_

#include "columns/BaseObject.hpp"
#include "components/WeightsPair.hpp"
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
    * @brief weightInitType: Specifies the type of weight initialization.
    * @details This parameter is not used directly inside the InitWeights class,
    * except to include the parameter in generated params files.
    * Generally, instantiation should proceed by separately reading the
    * WeightInitType and using the Factory::createByKeyword template to
    * instantiate the function.
    */
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);

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
    * @brief frameNumber: If initWeightsFile is set, the frameNumber parameter
    * selects which frame of the pvp file to use. The default value is zero.
    * Note that this parameter is zero-indexed: for example, if a pvp file
    * has five frames, the allowable values of this parameter are 0 through 4,
    * inclusive.
    */
   virtual void ioParam_frameNumber(enum ParamsIOFlag ioFlag);

   // useListOfArborFiles, combineWeightFiles, and numWeightFiles were marked obsolete July 13,
   // 2017.
   /**
    * @brief useListOfArborFiles is obsolete.
    */
   virtual void ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag);

   /**
    * @brief combineWeightFiles is obsolete.
    */
   virtual void ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag);

   /**
    * @brief numWeightFiles is obsolete.
    */
   virtual void ioParam_numWeightFiles(enum ParamsIOFlag ioFlag) {}
   /** @} */

  public:
   InitWeights(char const *name, HyPerCol *hc);
   virtual ~InitWeights();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   InitWeights();
   int initialize(const char *name, HyPerCol *hc);
   void handleObsoleteFlag(std::string const &flagName);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /*
    * initializeState checks initWeightsFile. If initWeightsFile is set, the weights are
    * loaded from the file. If it is not set, it calls calcWeights with no arguments.
    * Generally, derived classes should not override initializeState, but instead override
    * one or both of the calcWeights methods, in order to preserve the initWeightsFile behavior.
    */
   virtual Response::Status initializeState() override;

   /**
    * Called by initializeWeights, to calculate the weights in all arbors and all patches.
    * The base implementation callse calcWeights(int, int) in a loop over arbors and
    * patches
    */
   virtual void calcWeights();

   /**
    * Called by calcWeights(void), to calculate the weights in the given arbor and patch.
    * Derived classes generally override this method.
    */
   virtual void calcWeights(int dataPatchIndex, int arborId);

   virtual int readWeights(const char *filename, int frameNumber, double *timestampPtr = nullptr);

   virtual int initRNGs(bool isKernel) { return PV_SUCCESS; }

   int
   dataIndexToUnitCellIndex(int dataIndex, int *kx = nullptr, int *ky = nullptr, int *kf = nullptr);
   int kernelIndexCalculations(int patchIndex);
   float calcYDelta(int jPost);
   float calcXDelta(int iPost);
   float calcDelta(int post, float dPost, float distHeadPreUnits);

  protected:
   Weights *mWeights = nullptr; // initializeWeights sets this to the WeightsPair's PreWeights.

   char *mWeightInitTypeString = nullptr;

   char *mFilename  = nullptr;
   int mFrameNumber = 0;
   float mDxPost;
   float mDyPost;
   float mXDistHeadPreUnits;
   float mYDistHeadPreUnits;

}; // class InitWeights

} /* namespace PV */
#endif /* INITWEIGHTS_HPP_ */
