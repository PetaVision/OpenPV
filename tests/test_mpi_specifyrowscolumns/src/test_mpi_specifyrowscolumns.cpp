/**
 * This file tests whether the HyPerCol parses
 * the -rows and -columns arguments correctly
 * Use with mpirun -np 6
 *
 */

#include <include/pv_common.h>

#ifndef PV_USE_MPI
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[]) {
   pvError().printf("%s: this test can only be used under MPI with exactly six processes.\n", argv[0]);
   // TODO Greater than six should be permissible, with the excess over 6 being idle
}
#else // ifndef PV_USE_MPI

#include "columns/HyPerCol.hpp"
#include "layers/ANNLayer.hpp"
#include "io/io.hpp"
#include "assert.h"
#include "arch/mpi/mpi.h"

int buildandverify(PV::PV_Init* initObj);
int verifyLoc(PV::HyPerCol * loc, int rows, int columns);
int dumpLoc(const PVLayerLoc * loc, int rank);

using namespace PV;

int main(int argc, char * argv[]) {
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   int status = PV_SUCCESS;
   int rank = 0;
   int numProcs = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   //int numProcs = initObj->getWorldSize();

   if( numProcs != 6) {
      // TODO Greater than six should be permissible, with the excess over 6 being idle
      if (rank==0) {
         pvErrorNoExit().printf("%s: this test can only be used under MPI with exactly six processes.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   
   if (initObj->getParamsFile()!=NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the params file argument.\n", initObj->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (initObj->getNumRows()!=0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the rows argument.\n", initObj->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (initObj->getNumColumns()!=0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the columns argument.\n", initObj->getProgramName());
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         pvErrorNoExit().printf("The necessary parameters are hardcoded.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   initObj->setParams("input/test_mpi_specifyrowscolumns.params");
   initObj->setMPIConfiguration(2/*numRows*/, 3/*numColumns*/, -1/*batchWidth unchanged*/);
   buildandverify(initObj);

   initObj->setMPIConfiguration(3/*numRows*/, 2/*numColumns*/, -1/*batchWidth unchanged*/);
   buildandverify(initObj);

   delete initObj;
   return status;
}

int buildandverify(PV::PV_Init* initObj) {
   PV::HyPerCol * hc = new PV::HyPerCol("column", initObj);
   /* PV::ANNLayer * layer = */ new PV::ANNLayer("layer", hc);
   int rows = initObj->getNumRows();
   int columns = initObj->getNumColumns();
   pvErrorIf(!(rows > 0 && columns > 0), "Test failed.\n");
   int status = verifyLoc(hc, rows, columns);
   delete hc;
   return status;
}

int verifyLoc(PV::HyPerCol * hc, int rows, int columns) {
   int status = PV_SUCCESS;
   int testpassed;
   const PVLayerLoc * loc = hc->getLayer(0)->getLayerLoc();
   int rank = hc->getCommunicator()->commRank();
   pvErrorIf(!(rows == hc->getCommunicator()->numCommRows()), "Test failed.\n");
   pvErrorIf(!(columns == hc->getCommunicator()->numCommColumns()), "Test failed.\n");
   PVParams * params = hc->parameters();
   int nxGlobFromParams = params->value("column", "nx");
   int nyGlobFromParams = params->value("column", "ny");
   testpassed = (loc->nx == nxGlobFromParams/columns) &&
                (loc->ny == nyGlobFromParams/rows) &&
                (loc->nf == params->value("layer", "nf")) &&
                (loc->nxGlobal == nxGlobFromParams) &&
                (loc->nyGlobal == nyGlobFromParams) &&
                (loc->kx0 == loc->nx * (rank % columns)) &&
                (loc->ky0 == loc->ny * (rank / columns));

   PVLayerLoc mpiLoc;
   if( rank == 0 ) {
      pvInfo().printf("Testing with %d rows by %d columns of subprocesses.\n", rows, columns);
      if( testpassed ) {
         pvInfo().printf("Rank 0 passed.\n");
      }
      else {
         dumpLoc(loc, 0);
         pvErrorNoExit().printf("Rank 0 FAILED\n");
         status = PV_FAILURE;
      }
      // Receive each process's testpassed value and output it.
      for( int src=1; src<hc->getCommunicator()->commSize(); src++) {
         int remotepassed;
         MPI_Recv(&remotepassed, 1, MPI_INT, src, 10, hc->getCommunicator()->communicator(), MPI_STATUS_IGNORE);
         if( remotepassed ) {
            pvInfo().printf("Rank %d passed.\n", src);
         }
         else {
            MPI_Recv(&mpiLoc, sizeof(PVLayerLoc), MPI_CHAR, src, 20, hc->getCommunicator()->communicator(), MPI_STATUS_IGNORE);
            dumpLoc(&mpiLoc, src);
            pvErrorNoExit().printf("Rank %d FAILED\n", src);
            status = PV_FAILURE;
         }
      }
   }
   else {
      // Send each process's testpassed value to root process.
      MPI_Send(&testpassed, 1, MPI_INT, 0, 10, hc->getCommunicator()->communicator());
      if( !testpassed ) {
         memcpy(&mpiLoc, loc, sizeof(PVLayerLoc));
         MPI_Send(&mpiLoc, sizeof(PVLayerLoc), MPI_CHAR, 0, 20, hc->getCommunicator()->communicator());
      }
   }
   pvErrorIf(!(status == PV_SUCCESS), "Test failed.\n");
   return status;
}

int dumpLoc(const PVLayerLoc * loc, int rank) {
   if( loc == NULL ) return PV_FAILURE;
   pvInfo().printf("Rank %d: nx=%d, ny=%d, nf=%d, nxGlobal=%d, nyGlobal=%d, kx0=%d, ky0=%d\n",
          rank, loc->nx, loc->ny, loc->nf, loc->nxGlobal, loc->nyGlobal, loc->kx0, loc->ky0);
   return PV_SUCCESS;
}
#endif // PV_USE_MPI
