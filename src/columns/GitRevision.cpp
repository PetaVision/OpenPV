#include "GitRevision.hpp"
#include "pvGitRevision.h"

namespace PV {

std::string const GitRevision::print() {
   return mGitRevisionString;
}

std::string const GitRevision::mGitRevisionString = PV_GIT_REVISION;

} // namespace PV
