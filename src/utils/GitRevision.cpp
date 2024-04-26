#include "GitRevision.hpp"
#include "GitRevisionString.hpp"

namespace PV {

char const *getGitRevision() {
   return GitRevisionString<0>::getString();
}

} //namespace PV
