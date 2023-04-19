#ifndef PROBEDATA_HPP_
#define PROBEDATA_HPP_

#include "utils/PVAssert.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace PV {

template <typename T>
class ProbeData {
  public:
   typedef typename std::vector<T>::size_type size_type;
   ProbeData(double timestamp, size_type batchWidth, T initialValue);
   ProbeData(double timestamp, size_type batchWidth);
   ~ProbeData() {}

   size_type size() const { return mValues.size(); }

   static unsigned int calcPackedSize(unsigned int batchWidth);
   std::vector<char> pack() const;
   static ProbeData<T> unpack(std::vector<char> const &packedData);

   void reset(double timestamp);
   void reset(double timestamp, std::vector<double> const &newValues);
   void reset(ProbeData const &newValues);

   double getTimestamp() const { return mTimestamp; }

   T &getValue(int index) { return mValues.at(index); }
   T const &getValue(int index) const { return mValues.at(index); }
   std::vector<T> const &getValues() const { return mValues; }

  private:
   double mTimestamp;
   std::vector<T> mValues;
};

template <typename T>
ProbeData<T>::ProbeData(double timestamp, size_type batchWidth, T initialValue) {
   mTimestamp = timestamp;
   mValues.resize(batchWidth, initialValue);
}

template <typename T>
ProbeData<T>::ProbeData(double timestamp, size_type batchWidth) {
   mTimestamp = timestamp;
   mValues.resize(batchWidth);
}

template <typename T>
unsigned int ProbeData<T>::calcPackedSize(unsigned int batchWidth) {
   return sizeof(double) + batchWidth * sizeof(T);
}

template <typename T>
void ProbeData<T>::reset(double timestamp) {
   mTimestamp = timestamp;
   for (auto &x : mValues) {
      x = T();
   }
}

template <typename T>
void ProbeData<T>::reset(double timestamp, std::vector<double> const &newValues) {
   pvAssert(newValues.size() == mValues.size());
   mTimestamp = timestamp;
   std::copy(newValues.begin(), newValues.end(), mValues.begin());
}

template <typename T>
void ProbeData<T>::reset(ProbeData<T> const &newValues) {
   reset(newValues.getTimestamp(), newValues.getValues());
}

template <typename T>
std::vector<char> ProbeData<T>::pack() const {
   unsigned int dataSize     = size();
   unsigned int packedLength = sizeof(double) + dataSize * sizeof(T);
   std::vector<char> result(calcPackedSize(dataSize));
   char *position = &result.at(0);
   memcpy(position, &mTimestamp, sizeof(double));
   position += sizeof(double);
   for (unsigned int k = 0; k < dataSize; ++k) {
      memcpy(position, &mValues.at(k), sizeof(T));
      position += sizeof(T);
   }
   pvAssert(position - &result.at(0) == result.size());
   return result;
}

template <typename T>
ProbeData<T> ProbeData<T>::unpack(std::vector<char> const &packedData) {
   if (packedData.size() < sizeof(double) + sizeof(T)) {
      throw std::runtime_error("ProbeData::unpack() argument is too short.");
   }
   double timestamp;
   char const *position = &packedData.at(0);
   memcpy(&timestamp, position, sizeof(double));
   position += sizeof(double);
   unsigned int nbatch = (packedData.size() - sizeof(double)) / sizeof(T);
   if (packedData.size() != calcPackedSize(nbatch)) {
      throw std::runtime_error(
            "ProbeData::unpack() argument length does not match header and type.");
   }
   ProbeData<T> result(timestamp, nbatch);
   for (unsigned int k = 0U; k < nbatch; ++k) {
      T &value = result.getValue(k);
      memcpy(&value, position, sizeof(T));
      position += sizeof(T);
   }
   return result;
}

} // end namespace PV

#endif // PROBEDATA_HPP_
