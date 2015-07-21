/*
 * HeliTileMovie.hpp
 * Author: slundquist
 */

#ifndef HELITILEMOVIE_HPP_ 
#define HELITILEMOVIE_HPP_ 
#include <layers/Movie.hpp>
#include <string>
#include <fstream>
#include <iostream>

namespace PV{

class HeliTileMovie : public PV::Movie{
public:
   HeliTileMovie(const char * name, HyPerCol * hc);
   virtual ~HeliTileMovie();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_TargetCSV(enum ParamsIOFlag ioFlag);
   void ioParam_DistractorCSV(enum ParamsIOFlag ioFlag);
   void ioParam_CSVScale(enum ParamsIOFlag ioFlag);
   void ioParam_CSVOffset(enum ParamsIOFlag ioFlag);
   void ioParam_TimestampFile(enum ParamsIOFlag ioFlag);
   int ioParam_offsets(enum ParamsIOFlag ioFlag);

   int getGroundTruth(){return groundTruth[allPosIdx-1];}

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);
protected: 
   HeliTileMovie();
private:
   int initialize_base();
   int getCSVFrame(std::string csvString);
   std::pair <int, int> getCSVOffset(std::string csvString);
   float getTimestampTime(std::string timestampString);
   std::string getTimestampClipFrame(std::string timestampString);
   std::string getCSVClipFrame(std::string csvString);

   char * targetCSVfilename;
   char * distractorCSVfilename;
   char * timestampFilename;
   std::ifstream targetCSV;
   std::ifstream distractorCSV;
   std::ifstream timestampFile;
   std::string targetString;
   std::string distractorString;
   std::string timestampString;
   std::vector <std::pair <int, int> > allPos; //The vector of offsets to use
   std::vector <int> groundTruth; //The vector of ground truths corresponding to allPos
   int allPosIdx; //The current index the movie is on
   float csvXScale;
   float csvYScale;
   float csvXOffset;
   float csvYOffset;
   float timestampDisplayPeriod;
   bool newFrameNeeded;
};

}
#endif 
