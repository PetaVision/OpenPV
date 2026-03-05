#include <cstdlib>
#include <string>
#include <fstream>
#include "findMemoryUsed.hpp"
#include "utils/PVLog.hpp"

namespace PV {

   long int findMemoryUsed() {
      std::string line;
      std::string file;
      std::ifstream statusStream("/proc/self/status");
      if (!statusStream) {
         return -1L;
      }
      while (std::getline(statusStream, line)) {
         file.append(line).append("\n");
         if (line.size() >= 6UL and line.substr(0, 6) == "VmRSS:") {
            long int result = std::strtol(&line[6], nullptr, 10);
            return result;
         }
      }
      InfoLog() << file << "\n";
      return -1L;
   }

} // namespace PV
