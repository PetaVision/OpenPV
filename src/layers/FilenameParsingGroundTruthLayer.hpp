/*
 * FilenameParsingGroundTruthLayer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */
#ifndef FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
#define FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_

#include "HyPerLayer.hpp"
#include "InputLayer.hpp"
#include <string>

namespace PV {

class FilenameParsingGroundTruthLayer : public HyPerLayer {

  public:
   FilenameParsingGroundTruthLayer(const char *name, HyPerCol *hc);
   virtual ~FilenameParsingGroundTruthLayer();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int updateState(double timef, double dt) override;
   virtual bool needUpdate(double time, double dt) override;
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  private:
   std::vector<std::string> mClasses;
   char *mInputLayerName    = nullptr;
   char *mClassListFileName = nullptr;
   InputLayer *mInputLayer  = nullptr;
   float mGtClassTrueValue  = 1.0f;
   float mGtClassFalseValue = 0.0f;

  protected:
   virtual int registerData(Checkpointer *checkpointer) override;

   /**
    * List of protected paramters needed from FilenameParsingGroundTruthLayer
    * @name FilenameParsingGroundTruthLayer Paramters
    * @{
    */

   /**
    * @brief clasList: path to the .txt file that holds the list of imageListPath features
    * that will parse to different classifications
    * @details If this is not specified, the layer will attempt to use "classes.txt" in the output
    * directory. The identifying strings must be separated by a new line, and mutually exclusive.
    * In the case of CIFAR images, the pictures are organized in folders /0/ /1/ /2/ ... etc,
    * therefore those are the classifers
    * When an image is passed to the movie layer, the classification is parsed and a corresponding
    * neuron is activated to a value set by
    * gtClassTrueValue and the remaining neurons are set to the value set by gtClassFalseValue
    */

   virtual void ioParam_classList(enum ParamsIOFlag ioFlag);

   /**
    * @brief movieLayerName: lists name of the movie layer from which the imageListPath is used to
    * parse the classification
    */

   virtual void ioParam_inputLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief gtClassTrueValue: defines value to be set for the neuron that matches classes.txt
    * classifer
    * @details Default: 1
    */

   virtual void ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag);

   /**
    * @brief gtClassFalseValue: defines value to be set for the neurons that do not match the
    * classes.txt classifer
    * @details Default: -1
    */
   virtual void ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag);

   /** @} */
};
}

#endif // FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
