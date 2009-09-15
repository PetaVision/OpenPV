/*
 * HyPerCol.h
 *
 *  Created on: Jul 30, 2008
 *      Author: rasmussn
 */

#ifndef HYPERCOL_HPP_
#define HYPERCOL_HPP_

#include "../layers/PVLayer.h"
#include "../connections/HyPerConn.hpp"
#include "../io/PVParams.hpp"
#include "../include/pv_types.h"

namespace PV {

class HyPerLayer;
class InterColComm;
class HyPerConn;

class HyPerCol {

public:

   HyPerCol(const char* name, int argc, char* argv[]);
   virtual ~HyPerCol();

   int initFinish(void); // call after all layers/connections have been added
   int initializeThreads();
   int finalizeThreads();

   int run(int nTimeSteps);
   int run_old(int nTimeSteps); // has thread usage that needs to be moved
   int run()            {return run(numSteps);}

   int loadState();
   int writeState();

   int columnId();

//   int deliver(PVConnection* conn, PVRect preRegion, int count, float* buf);

   int addLayer(HyPerLayer * l);
   int addConnection(HyPerConn * conn);

   HyPerLayer * getLayer(int which)       {return layers[which];}
   HyPerConn  * getConnection(int which)  {return connections[which];}

   InterColComm * icCommunicator()        {return icComm;}

   PVParams * parameters()                {return params;}

   bool  warmStartup()                    {return warmStart;}

   float getDeltaTime()                   {return deltaTime;}
   float simulationTime()                 {return time;}

   LayerLoc getImageLoc()                 {return imageLoc;}
   float  width()                         {return imageLoc.nx;}
   float  height()                        {return imageLoc.ny;}

   const char * inputFile()               {return image_file;}

   int numberOfTimeSteps()                {return numSteps;}

   int numberOfColumns();

   int numberOfConnections()              {return numConnections;}

   /** returns the number of border regions, either an actual image border or a neighbor **/
   int numberOfBorderRegions()            {return MAX_NEIGHBORS;}

   int commColumn(int colId);
   int commRow(int colId);

private:
   int numSteps;
   int maxLayers;
   int numLayers;
   int maxConnections;
   int numConnections;

   bool warmStart;

   float time;                  // current time in milliseconds
   float deltaTime;             // time step interval

   HyPerLayer ** layers;
   HyPerConn  ** connections;

   int numThreads;
   PVLayer* threadCLayers;

   char * name;
   char * image_file;
   LayerLoc imageLoc;

   PVParams     * params; // manages input parameters
   InterColComm * icComm; // manages communication between HyPerColumns};

}; // class HyPerCol

} // namespace PV

extern "C" {
void *run1connection(void * arg); // generic prototype suitable for fork() : actually takes a run_struct
void *update1layer(void * arg); // generic prototype suitable for fork() : actually takes a run_struct
};

#endif /* HYPERCOL_HPP_ */
