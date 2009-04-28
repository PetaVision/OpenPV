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
   const char * inputFile()              {return input_file;}

//   int deliver(PVConnection* conn, PVRect preRegion, int count, float* buf);

   int addLayer(HyPerLayer * l);
   int addConnection(HyPerConn * conn);

   HyPerLayer * getLayer(int which)       {return layers[which];}
   HyPerConn  * getConnection(int which)  {return connections[which];}

   PVParams * parameters()                {return params;}

   float getDeltaTime()                   {return deltaTime;}
   float simulationTime()                 {return time;}

   PVRect getImageRect()                  {return imageRect;}
   float  width()                         {return imageRect.width;}
   float  height()                        {return imageRect.height;}

   int numberOfTimeSteps()                {return numSteps;}

   int numberOfColumns();

   int numberOfConnections()              {return numConnections;}

   int commColumn(int colId);
   int commRow(int colId);

private:
   int numSteps;
   int maxLayers;
   int numLayers;
   int maxConnections;
   int numConnections;

   float time;                  // current time in milliseconds
   float deltaTime;             // time step interval

   PVRect imageRect;
   HyPerLayer ** layers;
   HyPerConn  ** connections;

   int numThreads;
   PVLayer* threadCLayers;

   char * name;
   char * input_file;

   PVParams     * params; // manages input parameters
   InterColComm * icComm; // manages communication between HyPerColumns};

}; // class HyPerCol

} // namespace PV

// TODO - move thread stuff somewhere else

typedef struct run_struct_ {
   PV::HyPerCol *hc;
   int layer;
   int proc;
} run_struct;

extern "C" {
void *run1connection(void * arg); // generic prototype suitable for fork() : actually takes a run_struct
void *update1layer(void * arg); // generic prototype suitable for fork() : actually takes a run_struct
};

#endif /* HYPERCOL_HPP_ */
