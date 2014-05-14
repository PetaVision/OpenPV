/*
 * HeliGTLayer.cpp
 * Author: slundquist
 */

#include "HeliGTLayer.hpp"
#include <assert.h>
#include <dirent.h>

namespace PV {

HeliGTLayer::HeliGTLayer()
{
   initialize_base();
}

HeliGTLayer::HeliGTLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

HeliGTLayer::~HeliGTLayer(){
   if(timestampfile.is_open()) timestampfile.close();
   if(csvfile.is_open()) csvfile.close();
}

int HeliGTLayer::initialize_base(){
   currClip = "";
}

int HeliGTLayer::initialize(const char * name, HyPerCol * hc) {
   //TODO make only root process do this
   int status = ANNLayer::initialize(name, hc);

   if (parent->columnId()==0) {
      timestampfile.open(timestampFilename, std::ifstream::in);
      if (!timestampfile.is_open()){
         std::cout << "Unable to open file " << timestampFilename << "\n";
         exit(EXIT_FAILURE);
      }

      if(startFrame < 1){
         std::cout << "Setting startFrame to 1\n";
         startFrame = 1;
      }
      //Skip for startFrame
      for(int i = 0; i < startFrame; i++){
         getline (timestampfile,timestampString);
      }
   }
   return status;
}

int HeliGTLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_dataDir(ioFlag);
   ioParam_timestampFilename(ioFlag);
   ioParam_StartFrame(ioFlag);
   return status;
}

void HeliGTLayer::ioParam_dataDir(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "dataDir", &dataDir);
}

void HeliGTLayer::ioParam_timestampFilename(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "timestampFilename", &timestampFilename);
}

void HeliGTLayer::ioParam_StartFrame(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "startFrame", &startFrame, startFrame);
}

//Updates csvfile stream and frame offset
//Input is the string of the clip to update to
void HeliGTLayer::updateClip(std::string clipStr){
   std::cout << "Updating to new csv file\n";
   if (csvfile.is_open()){
      csvfile.close();
   }
   //Find frame offset
   std::string imageDir = std::string(dataDir) + "/" + clipStr + "/";
   DIR* dir = opendir(imageDir.c_str());
   frameoffset = 9999999;
   while (dirent* pdir = readdir(dir)){
      std::string filename = std::string(pdir->d_name);
      //skip hidden files
      if(filename.at(0) == '.'){
         continue;
      }
      const char * cFrame = filename.substr(0, 5).c_str();
      if(atoi(cFrame) < frameoffset){
         frameoffset = atoi(cFrame);
      }
   }

   //Open new csv file
   std::string csvFilename = std::string(dataDir) + "/CSV/" + clipStr + ".csv";
   csvfile.open(csvFilename.c_str(), std::ifstream::in);
   if (!csvfile.is_open()){
      std::cout << "Unable to open file " << csvFilename << "\n";
      exit(EXIT_FAILURE);
   }
   //Read line to remove header line
   getline(csvfile, csvString);
}

void HeliGTLayer::setA(float val){
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->nb);
      A[nExt] = val;
   }
}

void HeliGTLayer::setA(std::vector<int> inPoints, float val){
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   for(int i = 0; i < inPoints.size(); i++){
      int globalRes = inPoints[i];
      //Convert to x and y indicies
      int globalX = kxPos(globalRes, loc->nxGlobal, loc->nyGlobal, loc->nf);
      int globalY = kyPos(globalRes, loc->nxGlobal, loc->nyGlobal, loc->nf);
      int f = featureIndex(globalRes, loc->nxGlobal, loc->nyGlobal, loc->nf);
      //Check if this value exists in this process
      if(globalX >= loc->kx0 && globalX < loc->kx0+loc->nx &&
         globalY >= loc->ky0 && globalY < loc->ky0+loc->ny){
         //Calculate extended local value
         int localX = globalX - loc->kx0;
         int localY = globalY - loc->ky0;
         int localRes = kIndex(localX, localY, f, loc->nx, loc->ny, loc->nf);
         int localExt = kIndexExtended(localRes, loc->nx, loc->ny, loc->nf, loc->nb);
         A[localExt] = val;
      }
   }
}

void HeliGTLayer::getBoundingPoints(std::vector <int> &outVec, std::vector <int> boxVals){
   //Box must have 8 values
   assert(boxVals.size() == 8);
   const PVLayerLoc * loc = getLayerLoc(); 
   int xmin = 99999999;
   int xmax = -9999999;
   int ymin = 99999999;
   int ymax = -9999999;
   //Find outer bounding box
   for(int i = 0; i < 8; i++){
      //x vals
      if(i % 2 == 0){
         if(xmin > boxVals[i]) xmin = boxVals[i];
         if(xmax < boxVals[i]) xmax = boxVals[i];
      }
      //y vals
      else{
         if(ymin > boxVals[i]) ymin = boxVals[i];
         if(ymax < boxVals[i]) ymax = boxVals[i];
      }
   }
   //Search through this outer bounding box to find all points in that box
   //Based off of http://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
   for(int yi = ymin; yi <= ymax; yi++){
      for(int xi = xmin; xi <= xmax; xi++){
         bool inBoxAnd = true;
         bool inBoxOr = false;
         //Checking 4 angles
         for(int ai = 0; ai < 4; ai++){
            int currx = boxVals[ai*2];
            int curry = boxVals[ai*2+1];
            int nextx = boxVals[((ai+1)*2) % 8];
            int nexty = boxVals[(((ai+1)*2)+1) % 8];
            //Set currently checked point as orgin
            currx -= xi;
            curry -= yi;
            nextx -= xi;
            nexty -= yi;
            //Do calculation to find sign
            int val = nextx*curry - currx*nexty;
            //If this value is 0, it's right on the line, so if all other points match up, this point is in box
            if(val != 0){ 
               bool sign = val > 0;
               inBoxAnd &= sign;
               inBoxOr |= sign;
            }
         }
         //Depending on cw or countercw, inBoxAnd is 1 or inBoxOr is 0
         if(inBoxAnd || !inBoxOr){
            //Current point is in bounding box
            for(int fi = 0; fi < loc->nf; fi++){
               //Note that this is restricted
               int ki = kIndex(xi, yi, fi, loc->nxGlobal, loc->nyGlobal, loc->nf);
               outVec.push_back(ki);
            }
         }
      }
   }
   //Checked all points
}

int HeliGTLayer::getCSVFrame(std::string csvString){
   unsigned found = csvString.find_first_of(",");
   std::string stringFrame = csvString.substr(0, found); 
   int csvFrame = atoi(stringFrame.c_str()) + frameoffset;
   return csvFrame;
}

int HeliGTLayer::updateState(double timef, double dt) {
   int hasAnnotations = 0;

   //Define vectors of target points and dcr points
   //Defined in pv global restricted linear space
   std::vector <int> targetPts;
   std::vector <int> dcrPts;
   targetPts.clear();
   dcrPts.clear();

   if (parent->columnId()==0) {
      pvdata_t * A = getCLayer()->activity->data;
      const PVLayerLoc * loc = getLayerLoc(); 
      getline (timestampfile,timestampString);
      //Parse out clip number
      unsigned found = timestampString.find_last_of("/");
      std::string clipStr = timestampString.substr(found-3, 3);
      //Parse out frame number
      std::string frameStr = timestampString.substr(found+1, 5);
      std::cout << "time: " << timef << " timestampString: " << timestampString << "\n";

      //Check if we're still reading the same csv file
      if(currClip.compare(clipStr) != 0){
         updateClip(clipStr);
         currClip = clipStr;
      }

      //Need to find all objects in current frame
      int csvFrame = -1; 
      int timestampFrame = atoi(frameStr.c_str());
      //We may start at any given frame, so skip csv lines until we get to the frame in the timestamp file
      while(timestampFrame > csvFrame){
         getline(csvfile, csvString);
         csvFrame = getCSVFrame(csvString);
      }
      //Heli annotates every 4th frame, so if this is the case (no bounding boxes in the frame), set activity to zero and continue
      if(timestampFrame < csvFrame){
         hasAnnotations = 0;
         setA(0);
      }
      else{
         hasAnnotations = 1;
      }
#ifdef PV_USE_MPI
      MPI_Bcast(&hasAnnotations, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
#endif
      if(!hasAnnotations){
         return PV_SUCCESS;
      }
      //csvString is now on the correct frame
      //Calcluate the xScale and yScale of this current layer
      float xScale = pow(.5, getXScale());
      float yScale = pow(.5, getYScale());

      while(timestampFrame == csvFrame){
         std::stringstream lineStream(csvString);
         std::string cell;
         //First value is frame info, already checked, so don't need
         std::getline(lineStream, cell, ',');
         //Next 8 values is the bounding box
         std::vector <int> boxVec;
         boxVec.clear();
         for(int i = 0; i < 8; i++){
            std::getline(lineStream, cell, ',');
            int boxVal = atoi(cell.c_str());
            //Even i is x values, odd i is y values
            if(i % 2 == 0){
               boxVal = round(boxVal * xScale);
               //Check boundary cases
               if(boxVal < 0){
                  boxVal = 0;
               }
               else if(boxVal >= loc->nxGlobal){
                  boxVal = loc->nxGlobal - 1;
               }
            }
            else{
               boxVal = round(boxVal * yScale);
               //Check boundary cases
               if(boxVal < 0){
                  boxVal = 0;
               }
               else if(boxVal >= loc->nyGlobal){
                  boxVal = loc->nyGlobal - 1;
               }
            }
            boxVec.push_back(boxVal);
         }
         //Next value is the type of object
         std::getline(lineStream, cell, ',');
         //Only checking "Car" and "DCR" areas
         if(cell.compare("Car") == 0){
            getBoundingPoints(targetPts, boxVec);
         }
         else if(cell.compare("DCR") == 0){
            getBoundingPoints(dcrPts, boxVec);
         }
         //Get next line
         getline(csvfile, csvString);
         csvFrame = getCSVFrame(csvString);
      }//End for all bounding boxes in this frame
      //Need to broadcast the 2 vectors to all processes, TODO
#ifdef PV_USE_MPI
      //Broadcast target points
      int targetSize = targetPts.size();
      int dcrSize = dcrPts.size();
      //Broadcast size
      MPI_Bcast(&targetSize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      MPI_Bcast(&dcrSize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      //Broadcast array
      MPI_Bcast(&targetPts[0], targetSize, MPI_INT, 0, parent->icCommunicator()->communicator());
      MPI_Bcast(&dcrPts[0], dcrSize, MPI_INT, 0, parent->icCommunicator()->communicator());
#endif
   }//End if rank==0
#ifdef PV_USE_MPI
   //If rank is not 0
   else{
      MPI_Bcast(&hasAnnotations, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
      if(hasAnnotations){
         int targetSize, dcrSize;
         //Receive size from root
         MPI_Bcast(&targetSize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
         MPI_Bcast(&dcrSize, 1, MPI_INT, 0, parent->icCommunicator()->communicator());
         //Reserve these points
         targetPts.reserve(targetSize);
         dcrPts.reserve(dcrSize);
         //Resize to reset interal size member variable
         targetPts.resize(targetSize);
         dcrPts.resize(dcrSize);
         //Receive vectors
         MPI_Bcast(&targetPts[0], targetSize, MPI_INT, 0, parent->icCommunicator()->communicator());
         MPI_Bcast(&dcrPts[0], dcrSize, MPI_INT, 0, parent->icCommunicator()->communicator());
      }
      else{
         setA(0);
         return PV_SUCCESS;
      }
   }
#endif

   //Set the values of the ground truth
   //Default to -1, or negative example
   setA(-1);
   //Set targets to 1
   setA(targetPts, 1);
   //Set dcr to 0
   setA(dcrPts, 0);
   return PV_SUCCESS;
}
}
