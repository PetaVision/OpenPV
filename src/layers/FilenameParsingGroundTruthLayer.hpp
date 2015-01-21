/*
 * FilenameParsingGroundTruthLayer.hpp
 *
 *  Created on: Nov 10, 2014
 *      Author: wchavez
 */

#ifndef FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_
#define FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_

#include "ANNLayer.hpp"
#include <string>
#include "Movie.hpp"
namespace PV {

class FilenameParsingGroundTruthLayer: public PV::ANNLayer {
public:
   FilenameParsingGroundTruthLayer(const char * name, HyPerCol * hc);
   virtual ~FilenameParsingGroundTruthLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
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
   virtual void ioParam_classes(enum ParamsIOFlag ioFlag);
   virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_gtClassTrueValue(enum ParamsIOFlag ioFlag);
   virtual void ioParam_gtClassFalseValue(enum ParamsIOFlag ioFlag);
};

} /* namespace PV */
#endif /* FILENAMEPARSINGGROUNDTRUTHLAYER_HPP_ */
