/*
 * HyPerCol.h
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#include "HyPerColRunDelegate.hpp"
#include "../layers/PVLayer.h"
#include "../connections/HyPerConn.hpp"
#include "../io/PVParams.hpp"
#include "../include/pv_types.h"
#include <time.h>

#include "../arch/opencl/CLDevice.hpp"

namespace PV {

class HyPerLayer;
class InterColComm;
class HyPerConn;

class HyPerCol {

public:

   HyPerCol(const char* name, int argc, char* argv[]);
   virtual ~HyPerCol();

   int initFinish(void); // call after all layers/connections have been added
   int initializeThreads(int device);
   int finalizeThreads();

   int run()  {return run(numSteps);}
   int run(int nTimeSteps);

   float advanceTime(float time);
   int   exitRunLoop(bool exitOnFinish);

   int loadState();
   int writeState();

   int columnId();

//   int deliver(PVConnection* conn, PVRect preRegion, int count, float* buf);

   int addLayer(HyPerLayer * l);
   int addConnection(HyPerConn * conn);

   HyPerLayer * getLayer(int which)       {return layers[which];}
   HyPerConn  * getConnection(int which)  {return connections[which];}

   CLDevice   * getCLDevice()             {return clDevice;}

   InterColComm * icCommunicator()        {return icComm;}

   PVParams * parameters()                {return params;}

   bool  warmStartup()                    {return warmStart;}

   float getDeltaTime()                   {return deltaTime;}
   float simulationTime()                 {return simTime;}

   PVLayerLoc getImageLoc()               {return imageLoc;}
   int width()                            {return imageLoc.nxGlobal;}
   int height()                           {return imageLoc.nyGlobal;}
   int localWidth()                       {return imageLoc.nx;}
   int localHeight()                      {return imageLoc.ny;}

   int setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int margin, int nf);

   const char * inputFile()               {return image_file;}

   int numberOfTimeSteps()                {return numSteps;}

   int numberOfColumns();

   int numberOfConnections()              {return numConnections;}

   /** returns the number of border regions, either an actual image border or a neighbor **/
   int numberOfBorderRegions()            {return MAX_NEIGHBORS;}

   int commColumn(int colId);
   int commRow(int colId);
   int numCommColumns()                   {return icComm->numCommColumns();}
   int numCommRows()                      {return icComm->numCommRows();}

   // a random seed based on column id
   unsigned long getRandomSeed()
      {return (unsigned long) time((time_t *) NULL) / (1 + columnId());}

   void setDelegate(HyPerColRunDelegate * delegate)  {runDelegate = delegate;}

private:
   int numSteps;
   int maxLayers;
   int numLayers;
   int maxConnections;
   int numConnections;

   bool warmStart;
   bool isInitialized;     // true when all initialization has been completed

   float simTime;          // current time in milliseconds
   float deltaTime;        // time step interval

   CLDevice * clDevice;    // object for running kernels on OpenCL device

   HyPerLayer ** layers;
   HyPerConn  ** connections;

   char * name;
   char * image_file;
   PVLayerLoc imageLoc;

   PVParams     * params; // manages input parameters
   InterColComm * icComm; // manages communication between HyPerColumns};

   HyPerColRunDelegate * runDelegate; // runs time loop

}; // class HyPerCol

} // namespace PV

#endif /* HYPERCOL_HPP_ */
