#include "CLDevice.hpp"

#ifdef PV_USE_OPENCL
int main(int argc, char * argv[])
{

   PV::CLDevice * cld = new PV::CLDevice(CL_DEVICE_DEFAULT);

   // query and print information about the devices found
   //
   cld->query_device_info();

   return 0;
}
#endif // PV_USE_OPENCL
