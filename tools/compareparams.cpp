/**
 * compareparams, a C++ program to compare the contents of two params files.
 * Usage: compareparams file1.params file2.params
 */

#include <columns/PV_Init.hpp>
#include <utils/CompareParamsFiles.hpp>

#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE
#include <cstring> // strdup
#include <getopt.h> // getopt_long
#include <libgen.h> // strdup
#include <unistd.h> // optind, used by getopt_long
#include <vector> // vector of long options

void showHelpMessage(std::string const &progName);

int main(int argc, char *argv[]) {
   // Get base name of program path, for use in print-statements.
   char *progPath       = strdup(argv[0]);
   std::string progName = basename(progPath);
   free(progPath);

   // Initialize PetaVision and get the Communicator.
   // Even though we run on a single process, we need the communicator because the PVParams
   // constructor requires it.
   int initargc    = 1;
   char **initargs = (char **)calloc((std::size_t)2, sizeof(char *));
   initargs[0]     = strdup(progName.c_str());
   initargs[1]     = nullptr;
   auto *pvInitObj = new PV::PV_Init(&initargc, &initargs, false /*no unrecognized args*/);
   free(initargs[0]);
   free(initargs);
   initargs = nullptr;

   bool error   = false;
   int showHelp = 0;
   vector<struct option> longopts{
         {"help", 0, &showHelp, 1}, {"usage", 0, &showHelp, 1}, {nullptr, 0, nullptr, 0}};
   int result = 0;
   while (result != -1) {
      result = getopt_long(argc, argv, "hu", longopts.data(), &optind);
      switch (result) {
         case (int)'h': showHelp = 1; break;
         case (int)'u': showHelp = 1; break;
         case -1: break;
         case 0: break;
         default:
            error = true; // getopt_long() produces the relevant error message.
      }
   }
   if (error) {
      return EXIT_FAILURE;
   }

   // Was the help message requested?
   if (showHelp) {
      showHelpMessage(progName);
      return EXIT_SUCCESS;
   }

   // Require exactly two filenames
   if (optind + 2 != argc) {
      std::cerr << progName << " requires exactly two arguments (" << argc - optind
                << " were given).\n";
      return EXIT_FAILURE;
   }

   std::string file1 = argv[optind];
   std::string file2 = argv[optind + 1];
   auto *comm        = pvInitObj->getCommunicator();

   std::stringstream configStream("NumRows:0\nNumColumns:0\nBatchWidth:0\n");

   int status = PV::compareParamsFiles(file1, file2, comm);
   delete pvInitObj;
   if (status == PV_SUCCESS) {
      std::cout << file1 << " and " << file2 << " are equivalent.\n";
   }
   else {
      return EXIT_FAILURE;
   }
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void showHelpMessage(std::string const &progName) {
   std::cout << progName << " [options]\n";
   std::cout << progName << " file1.params file2.params\n";
   std::cout << "\n";
   std::cout << "Options:\n";
   std::cout << "    -h, --help, -u, --usage\n";
   std::cout << "        Display this help text and exit. No other arguments are processed.\n";
   std::cout << "\n";
   std::cout << "The program takes two input paths, which should be PetaVision .params files.\n";
   std::cout << "It compares them, irrespective of ordering of parameter groups or of \n";
   std::cout << "ordering within a group, and outputs the differences.\n";
}
