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

//   int rank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   if(rank == 0){
//      printf("Hit enter to begin! ");
//      fflush(stdout);
//      int charhit = -1;
//      while(charhit != '\n') {
//         charhit = getc(stdin);
//      }
//   }
//#ifdef PV_USE_MPI
//   MPI_Barrier(MPI_COMM_WORLD);
//#endif // PV_USE_MPI
}

PV_Init::~PV_Init(){
   if(params){
      delete params;
   }
   //if(icComm){
   //   delete icComm;
   //}
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

   icComm = new InterColComm(argc, argv);
   //Read parameters and send to everyone
   params = new PVParams(param_file, 2*(INITIAL_LAYER_ARRAY_SIZE+INITIAL_CONNECTION_ARRAY_SIZE), icComm);
   //Get nbatch dimension from params
   //nbatch = (int) params->value(group_name, param_name, defaultValue, warnIfAbsent);
   //Set up communicators
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
   return 0;
}

int PV_Init::commInit(int* argc, char*** argv)
{
#ifdef PV_USE_MPI
   // If MPI wasn't initialized, initialize it.
   // Remember if it was initialized on entry; the destructor will only finalize if the constructor init'ed.
   // This way, you can do several simulations sequentially by initializing MPI before creating
   // the first HyPerCol; after running the first simulation the MPI environment will still exist and you
   // can run the second simulation, etc.
   MPI_Initialized(&initialized);
   if( !initialized) {
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



