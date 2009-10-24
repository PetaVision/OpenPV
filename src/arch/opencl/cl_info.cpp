#include "../CLDevice.hpp"

int main(int argc, char * argv[])
{
   PV::CLDevice * cld = new PV::CLDevice();

   // query and print information about the devices found
   //
   cld->query_device_info();

   return 0;
}
