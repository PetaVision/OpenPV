#ifndef PARAMETERSWEEP_HPP_
#define PARAMETERSWEEP_HPP_

#include <string>
#include <vector>

namespace PV {

enum SweepType { SWEEP_UNDEF = 0, SWEEP_NUMBER = 1, SWEEP_STRING = 2 };

class ParameterSweep {
  public:
   ParameterSweep();
   virtual ~ParameterSweep();

   int setGroupAndParameter(std::string const &groupName, std::string const &paramName);
   int pushNumericValue(double val);
   int pushStringValue(std::string const &sval);
   int getNumValues() const { return mNumValues; }
   SweepType getType() const { return mType; }
   int getNumericValue(int n, double *val);
   int getStringValue(int n, char const **sval);
   std::string const &getGroupName() const { return mGroupName; }
   std::string const &getParamName() const { return mParamName; }

  private:
   std::string mGroupName;
   std::string mParamName;
   SweepType mType = SWEEP_UNDEF;
   int mNumValues = 0;
   std::vector<double> mValuesNumber;
   std::vector<std::string> mValuesString;
};

} // namespace PV

#endif // PARAMETERSWEEP_HPP_
