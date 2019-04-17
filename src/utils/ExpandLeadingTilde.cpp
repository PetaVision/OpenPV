#include "ExpandLeadingTilde.hpp"
#include <cstdlib> // getenv

namespace PV {

std::string expandLeadingTilde(std::string const &path) {
   std::string result("");
   if (path.empty()) {
      return result;
   }
   if (path == std::string("~")) {
      result = getHomeDirectory();
   }
   else if (path.at(0) == '~' and path.at(1) == '/') {
      result = getHomeDirectory() + '/' + path.substr(2, path.size() - 2);
   }
   else {
      result = path;
   }
   return result;
}

std::string expandLeadingTilde(char const *path) { return expandLeadingTilde(std::string(path)); }

std::string const getHomeDirectory() {
   char const *homeDir = std::getenv("HOME");
   return std::string(homeDir ? homeDir : "");
}

} // end namespace PV
