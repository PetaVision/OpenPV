/*
 * HeliGTLayer.hpp
 * Author: slundquist
 */

#ifndef HELIGTLAYER_HPP_ 
#define HELIGTLAYER_HPP_ 
#include <layers/ANNLayer.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace PV{

class HeliGTLayer : public PV::ANNLayer{
public:
   HeliGTLayer(const char * name, HyPerCol * hc);
   virtual ~HeliGTLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_timestampFilename(enum ParamsIOFlag ioFlag);
   void ioParam_dataDir(enum ParamsIOFlag ioFlag);
   void ioParam_StartFrame(enum ParamsIOFlag ioFlag);
protected: 
   HeliGTLayer();
private:
   void updateClip(std::string clipStr);
   void getBoundingPoints(std::vector <int> &outVec, std::vector <int> boxVals);
   int getCSVFrame(std::string csvString);
   void setA(float val);
   void setA(std::vector<int> inPoints, float val);

   int initialize_base();
   std::string timestampString;
   std::string csvString;
   char* timestampFilename;
   char* dataDir;
   std::ifstream timestampfile;
   std::ifstream csvfile;
   long startFrame; //Zero indexed
   std::string currClip;
   int frameoffset;
};

}
#endif 
