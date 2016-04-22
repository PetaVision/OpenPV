/*
 * FilenameParsingGroundTruthLayer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#ifndef FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
#define FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_

#include <cMakeHeader.h>
#include "ANNLayer.hpp"
#include <string>
#include "Movie.hpp"
namespace PV {

class FilenameParsingGroundTruthLayer: public PV::ANNLayer {

#ifdef PV_USE_GDAL

public:
   FilenameParsingGroundTruthLayer(const char * name, HyPerCol * hc);
   virtual ~FilenameParsingGroundTruthLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int updateState(double timef, double dt);
   virtual bool needUpdate(double time, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
private:
   std::ifstream inputfile;
   std::string * classes;
   int numClasses;
   char * movieLayerName;
   Movie * movieLayer;
   float gtClassTrueValue;
   float gtClassFalseValue;
protected:

   /**
    * List of protected paramters needed from FilenameParsingGroundTruthLayer
    * @name FilenameParsingGroundTruthLayer Paramters
    * @{
    */
   
   /**
    * @brief classes: list the name of the .txt file that holds the list of imageListPath features that will parse to different classifications
    * @details classes.txt must be located in the output directory, the classifers separated by a new line, and must be discerning
    * In the case of CIFAR images, the pictures are organized in folders /0/ /1/ /2/ ... etc, therefore those are the classifers
    * When an image is passed to the movie layer, the classification is parsed and a corresponding neuron is activated to a value set by 
    * gtClassTrueValue and the remaining neurons are set to the value set by gtClassFalseValue
    */

   virtual void ioParam_classes(enum ParamsIOFlag ioFlag);

   /**
    * @brief movieLayerName: lists name of the movie layer from which the imageListPath is used to parse the classification
    */

   virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief gtClassTrueValue: defines value to be set for the neuron that matches classes.txt classifer
    * @details Default: 1
    */
   
   virtual void ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag);

   /**
    * @brief gtClassFalseValue: defines value to be set for the neurons that do not match the classes.txt classifer
    * @details Default: -1
    */
   virtual void ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag);

   /** @} */
#else // PV_USE_GDAL
public:
    FilenameParsingGroundTruthLayer(char const * name, HyPerCol * hc);
protected:
    FilenameParsingGroundTruthLayer();
#endif // PV_USE_GDAL
}; // class FilenameParsingGroundTruthLayer

BaseObject * createFilenameParsingGroundTruthLayer(char const * name, HyPerCol * hc);

} /* namespace PV */

#endif /* FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_ */
