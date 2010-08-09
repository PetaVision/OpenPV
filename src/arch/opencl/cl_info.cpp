#include "CLDevice.hpp"

int main(int argc, char * argv[])
{
#ifdef PV_USE_OPENCL

   PV::CLDevice * cld = new PV::CLDevice(CL_DEVICE_DEFAULT);

   // query and print information about the devices found
   //
   cld->query_device_info();

#endif // PV_USE_OPENCL

   return 0;
}
