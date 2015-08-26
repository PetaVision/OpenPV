/*
 * PV_Init.cpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */
#include "PV_Init.hpp"

namespace PV {

PV_Init::PV_Init(int* argc, char ** argv[]){
   //Initialize MPI
   commInit(argc, argv);
   params = NULL;
   icComm = NULL;
   initialized = false;

//      if( rank == 0 ) {
//         printf("Hit enter to begin! ");
//         fflush(stdout);
//         int charhit = -1;
//         while(charhit != '\n') {
//            charhit = getc(stdin);
//         }
//      }
//#ifdef PV_USE_MPI
//      MPI_Barrier(icComm->globalCommunicator());
//#endif // PV_USE_MPI


}

PV_Init::~PV_Init(){
   if(params){
      delete params;
   }
   if(icComm){
      delete icComm;
   }
   commFinalize();
}

int PV_Init::initialize(int argc, char* argv[]){
   if(icComm){
      delete icComm;
   }
   if(params){
      delete params;
   }
   //Parse param file
   char * param_file = NULL;
   pv_getopt_str(argc, argv, "-p", &param_file, NULL);

   if(!param_file){
      std::cout << "PV_Init setParams: initialize requires a -p parameter for a parameter file\n";
      exit(-1);
   }

   //Set up communicator and parameters
   icComm = new InterColComm(argc, argv);
   params = new PVParams(param_file, 2*(INITIAL_LAYER_ARRAY_SIZE+INITIAL_CONNECTION_ARRAY_SIZE), icComm);
   if(param_file){
      free(param_file);
   }
   initialized = true;
   return 0;
}

int PV_Init::initialize(PVParams* inparams, InterColComm* incomm){
   assert(inparams);
   assert(incomm);
   if(params){
      delete params;
   }
   if(icComm){
      delete icComm;
   }
   params = inparams;
   icComm = incomm;
   initialized = true;
   return 0;
}

int PV_Init::commInit(int* argc, char*** argv)
{
#ifdef PV_USE_MPI
   int mpiInit;
   // If MPI wasn't initialized, initialize it.
   // Remember if it was initialized on entry; the destructor will only finalize if the constructor init'ed.
   // This way, you can do several simulations sequentially by initializing MPI before creating
   // the first HyPerCol; after running the first simulation the MPI environment will still exist and you
   // can run the second simulation, etc.
   MPI_Initialized(&mpiInit);
   if( !mpiInit) {
      assert((*argv)[*argc]==NULL); // Open MPI 1.7 assumes this.
      MPI_Init(argc, argv);
   }
   else{
      std::cout << "Error: PV_Init communicator already initialized\n";
      exit(-1);
   }
#endif

   return 0;
}

int PV_Init::commFinalize()
{
#ifdef PV_USE_MPI
   MPI_Finalize();
#endif
   return 0;
}

}



