/*
 * LayerData.cpp
 *
 *  Created on: Jan 31, 2011
 *      Author: manghel
 */

#include "LayerData.hpp"
#include "../layers/LIF.cpp"

namespace PV {

//
/*
 * The default base constructor assigns stdout to the file pointer
 * - set extended flag for each type of state variable
 */
LayerData::LayerData(DataType data_type, bool append)
   : LayerProbe()
{
   this->data_type = data_type;
   if(data_type == TYPE_VTH){
      this->extended   = false;
   } else if(data_type == TYPE_WMAX){
      this->extended   = true;
   } else if (data_type == TYPE_R){
      this->extended   = false;
   } else if (data_type == TYPE_VTHREST){
      this->extended   = false;
   }

   this->append = append;
   if(!append){
      write_header = true;
   } else {
      write_header = false;
   }
   this->open_file = true;

   // reset fp to NULL here
   this->fp = NULL;
}


/**
 * @time
 * @l
 */
int LayerData::outputState(float time, HyPerLayer * l)
{
   bool contiguous = false;
   HyPerCol * parent = l->getParent();
   pvdata_t * data;

   int status = 0;
   Communicator * comm = parent->icCommunicator();
   PVLayerLoc * loc = & ((l->getCLayer())->loc);
   //float time = parent->simulationTime();

   LIF * LIF_layer = dynamic_cast<LIF *>(l);
   assert(LIF_layer != NULL);

   if(open_file){ // find file name, open file pointer, get data pointer, write header if !append

      char filename[PV_PATH_MAX];
      const char * last_str = "";
      int numNeurons = l->getCLayer()->numNeurons;

      if(data_type == TYPE_VTH){
         data = LIF_layer->getVth();
         l->getOutputFilename(filename, "Vth", last_str);
      } else if(data_type == TYPE_WMAX){
         data = LIF_layer->getWmax();
         l->getOutputFilename(filename, "Wmax", last_str);
      } else if (data_type == TYPE_R){
         data = LIF_layer->getR();
         l->getOutputFilename(filename, "R", last_str);
      } else if (data_type == TYPE_VTHREST){
         data = LIF_layer->getVthRest();
         l->getOutputFilename(filename, "VthRest", last_str);
      }

      this->fp = pvp_open_write_file(filename, comm, this->append);

      if(!append){
         status |= pvp_write_header(this->fp, comm, time, loc, PVP_FILE_TYPE,
               PV_FLOAT_TYPE, sizeof(int), extended, contiguous, NUM_BIN_PARAMS, numNeurons);
         assert(status == 0);
      }
      open_file = false;

   } else { // get data pointer

      if(data_type == TYPE_VTH){
         data = LIF_layer->getVth();
      } else if(data_type == TYPE_WMAX){
         data = LIF_layer->getWmax();
      } else if (data_type == TYPE_R){
         data = LIF_layer->getR();
      } else if (data_type == TYPE_VTHREST){
         data = LIF_layer->getVthRest();
      }
   }

   status |= write(fp, comm, time, data, loc, PV_FLOAT_TYPE, extended, contiguous);
   return status;

}

}
