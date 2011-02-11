/*
 * LayerData.cpp
 *
 *  Created on: Jan 31, 2011
 *      Author: manghel
 */

#include "LayerData.hpp"

namespace PV {

//
/*
 * The default base constructor assign NULL to the file pointer
 */
LayerData::LayerData(const char * filename, HyPerCol * hc, HyPerLayer * l, pvdata_t * data, bool append)
   : LayerProbe()
{
   this->data = data;
   this->parent = hc;
   this->append = append;
   Communicator * comm = parent->icCommunicator();

   this->fp = pvp_open_write_file(filename, comm, this->append);
   if(!append){
      bool contiguous = false;
      bool extended   = false;
      int status = 0;
      PVLayerLoc * loc = & ((l->getCLayer())->loc);
      float time = hc->simulationTime();
      int numNeurons = l->getCLayer()->numNeurons;

      status |= pvp_write_header(this->fp, comm, time, loc, PVP_FILE_TYPE,
            PV_FLOAT_TYPE, sizeof(int), extended, contiguous, NUM_BIN_PARAMS, numNeurons);
      assert(status == 0);
   }

}

/**
 * @time
 * @l
 */
int LayerData::outputState(float time, HyPerLayer * l)
{
   Communicator * comm = parent->icCommunicator();
   PVLayerLoc * loc = & ((l->getCLayer())->loc);
   bool contiguous = false;
   bool extended   = false;
   int status = 0;
//   int status = write(fp, comm, time, data, loc, PV_FLOAT_TYPE, extended, contiguous);
   return status;
}


}
