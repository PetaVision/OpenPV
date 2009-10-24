/*
 * CLDevice.hpp
 *
 *  Created on: Oct 24, 2009
 *      Author: rasmussn
 */

#ifndef CLDEVICE_HPP_
#define CLDEVICE_HPP_

#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// guess at maxinum number of devices (CPU + GPU)
//
#define MAX_DEVICES (2)

#define CL_GET_DEVICE_ID_FAILURE    1


////////////////////////////////////////////////////////////////////////////////

namespace PV {


class CLDevice {
public:
   CLDevice();
   virtual ~CLDevice();

   int initialize();

   int query_device_info();

protected:

   int query_device_info(int id, cl_device_id device);

   cl_uint num_devices;                  // number of computing devices

   cl_device_id device_ids[MAX_DEVICES]; // compute device id
   cl_context context;                   // compute context
   cl_command_queue commands;            // compute command queue
   cl_program program;                   // compute program
   cl_kernel kernel;                     // compute kernel

};

} // namespace PV

#endif /* CLDEVICE_HPP_ */
