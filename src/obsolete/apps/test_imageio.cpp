/*
 * test_imageio.cpp
 *
 *  Created on: Sep 3, 2009
 *      Author: rasmussn
 */


int main(int argc, char * argv[])
{
   int err = 0;
   LayerLoc loc;

   const char * filename = "/Users/rasmussn/Codes/PANN/"
                           "world.topo.200408.3x21600x21600.C2.jpg";

   PV::Communicator * comm = new PV::Communicator(&argc, &argv);

   loc.nx = 256;
   loc.ny = 256;
   loc.nPad = 16;

   getImageInfo(filename, comm, &loc);

   printf("[%d]: nx==%d ny==%d nxGlobal==%d nyGlobal==%d kx0==%d ky0==%d\n",
          comm->commRank(), (int)loc.nx, (int)loc.ny,
          (int)loc.nxGlobal, (int)loc.nyGlobal, (int)loc.kx0, (int)loc.ky0);

   int xSize = loc.nx  + 2 * loc.nPad;
   int ySize = loc.ny  + 2 * loc.nPad;

   float * buf = new float[xSize*ySize];

   //   err = scatterImageFile(filename, ic, &loc, buf);
   //   if (err) exit(err);

   err = scatterImageBlocks(filename, comm, &loc, buf);
   if (err) exit(err);

   delete buf;
   delete comm;

   return err;
}
