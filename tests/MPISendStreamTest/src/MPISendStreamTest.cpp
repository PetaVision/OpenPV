#include <columns/Arguments.hpp>
#include <include/pv_common.h>
#include <io/io.hpp>
#include <io/MPIRecvStream.hpp>
#include <io/MPISendStream.hpp>
#include <utils/PVLog.hpp>
#include <utils/WaitForReturn.hpp>

#include <cstdlib> // system()
#include <fstream>
#include <mpi.h>
#include <string>
#include <sys/stat.h>
// #include <time.h>

// This test checks the MPISendStream and MPIRecvStream classes.
// Each process creates the directory "output/rank[n]" where [n] is the process's global rank,
// and first removes any contents of that directory with system("rm -rf output/rank[n]/*"), to
// start with a clean slate. Each process then loops over all ranks. For its own rank, it writes
// a test message to a file. For the other ranks, it uses MPISendStream to send a test message to
// that rank. Then, each process loops over all ranks except itself, and uses MPIRecvStream to
// write the test message to a file.
//
// The path of each test message is "output/rank[receiver]/SendingProg[sender].txt
// The contents of each test message is
// "Testing with sending rank [sender], receiving rank [receiver]\n"
// Here [sender] and [receiver] are the global ranks of the sending process and receiving process.
// Each process uses "output/rank[n]/MPISendStreamTest.log" as its log file.
// Note that process [n] only works in the directory "output/rank[n]", so there should be
// no collisions.

std::string generateDir(int rank);
std::string generateOutputString(int recvRank, int sendRank);
std::string generatePath(int recvRank, int sendRank);
void makeOutputDir(std::string const &dirPath);
void runOnSameProcs(int rank);
void runReceive(int recvRank, int sendRank);
void runSend(int recvRank, int sendRank);

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv);
   auto arguments = PV::parse_arguments(argc, argv, false /*do not allow unrecognized arguments*/);

   if (arguments->getBooleanArgument("RequireReturn")) {
      PV::WaitForReturn(MPI_COMM_WORLD);
   }

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   for (int r = 0; r < size; ++r) {
      if (r == rank) { makeOutputDir(std::string("output")); }
      MPI_Barrier(MPI_COMM_WORLD);
   }
   std::string outputDir = generateDir(rank);
   makeOutputDir(outputDir);
   std::string rmstring("rm -rf ");
   rmstring.append(outputDir).append("/*");
   int rmrfstatus = std::system(rmstring.c_str());
   FatalIf(
         !WIFEXITED(rmrfstatus),
         "system command \"%s\" did not exit normally\n",
         rmstring.c_str());
   FatalIf(
         WEXITSTATUS(rmrfstatus) != 0, 
         "system command \"%s\" returned status %d instead of expected 0.\n",
         rmstring.c_str(),
         WEXITSTATUS(rmrfstatus));

   std::string logFile = arguments->getStringArgument(std::string("LogFile"));
   if (!logFile.empty()) {
      logFile = outputDir + '/' + logFile;
   }
   PV::setLogFile(logFile);
   
   // Create test files
   for (int recver = 0; recver < size; ++recver) {
      if (rank == recver) {
         runOnSameProcs(recver);
      }
      else {
         runSend(recver, rank);
      }
   }
   for (int sender = 0; sender < size; ++sender) {
      if (sender != rank) {
         runReceive(rank, sender);
      }
   }

   // Check test files
   int status = PV_SUCCESS;
   for (int recver = 0; recver < size; ++recver) {
      if (rank == recver) {
         for (int sender = 0; sender < size; ++sender) {
            std::string path = generatePath(recver, sender);
            std::ifstream file(path.c_str());
            file.seekg(0, std::ios_base::end);
            auto filelen = file.tellg();
            std::string fileContents(filelen, '\0');
            file.seekg(0, std::ios_base::beg);
            file.read(&fileContents.front(), filelen);

            std::string correctContents = generateOutputString(recver, sender);
            if (fileContents != correctContents) {
               ErrorLog().printf(
                     "Sender rank %d: expected \"%s\", received \"%s\"\n",
                     sender, correctContents.c_str(), fileContents.c_str());               
               status = PV_FAILURE;
            }
         }
      }
   }

   int msgPresent = 1;
   MPI_Status probeStatus;
   while(msgPresent) {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msgPresent, &probeStatus);
      if (msgPresent) {
         int count;
         MPI_Get_count(&probeStatus, MPI_Datatype MPI_CHAR, &count);
         int sender = probeStatus.MPI_SOURCE;
         int tag = probeStatus.MPI_TAG;
         int error = probeStatus.MPI_ERROR;
         InfoLog() << "Unretrieved message, sending rank " << sender << ", tag " << tag <<
                      ", error " << error << ", count " << count << "\n";
         std::string recvString(count, '\0');
         char *buffer = &recvString.front();
         MPI_Recv(buffer, count, MPI_CHAR, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   status = status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   MPI_Finalize();
   return status;
}

std::string generateOutputString(int recvRank, int sendRank) {
   std::string outputString("Testing with sending rank ");
   outputString.append(std::to_string(sendRank));
   outputString.append(", receiving rank ");
   outputString.append(std::to_string(recvRank));
   outputString.append("\n");
   return outputString;
}

std::string generateDir(int rank) {
   std::string path("output/rank");
   path.append(std::to_string(rank));
   return path;
}

std::string generatePath(int recvRank, int sendRank) {
   std::string path = generateDir(recvRank);
   path.append("/");
   path.append("SendingProc").append(std::to_string(sendRank));
   path.append(".txt");
   return path;
}

void makeOutputDir(std::string const &dirPath) {
   struct stat pathstat;
   int statresult = stat(dirPath.c_str(), &pathstat);
   if (statresult == 0) {
      FatalIf(
             !S_ISDIR(pathstat.st_mode),
             "Path \"%s\" exists but is not a directory\n",
             dirPath.c_str());
      return;
   }
   FatalIf(
         errno != ENOENT,
         "Checking status of directory \"%s\" gave error \"%s\".\n",
         dirPath.c_str(),
         strerror(errno));
   InfoLog() << "Making directory " << dirPath << std::endl;
   int mkdirstatus = mkdir(dirPath.c_str(), 0777);
   if (mkdirstatus != 0) {
      Fatal().printf(
            "Directory \"%s\" could not be created: %s; Exiting\n", dirPath.c_str(), strerror(errno));
   }
}

void runOnSameProcs(int rank) {
   auto outputString = generateOutputString(rank, rank);
   auto path = generatePath(rank, rank);
   std::ofstream file(path);
   file << outputString;
}

void runReceive(int recvRank, int sendRank) {
   auto path = generatePath(recvRank, sendRank);
   PV::MPIRecvStream stream(path, MPI_COMM_WORLD, sendRank, false /*clobberFlag*/);
   int count = 0;
   int numPolls = 0;
   while(count == 0) {
      count = stream.receive(400 + recvRank + 20 * sendRank);
      ++numPolls;
   }
   InfoLog() << "MPIRecvStream::receive() from rank " << sendRank <<
                " needed to be called " << numPolls << " times. " <<
                count << " characters were received.\n";
}

void runSend(int recvRank, int sendRank) {
   PV::MPISendStream stream(MPI_COMM_WORLD, recvRank);
   auto contents = generateOutputString(recvRank, sendRank);
   stream << contents;
   int count = stream.send(400 + recvRank + 20 * sendRank);
   InfoLog() << "MPISendStream::send() to rank " << recvRank << ". " <<
                count << " characters were sent.\n";
}
