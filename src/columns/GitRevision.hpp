#include <string>

namespace PV {

class GitRevision {
  public:
   GitRevision() {}
   ~GitRevision() {}
   static std::string const print();

  private:
   static std::string const mGitRevisionString;
};

} // namespace PV
