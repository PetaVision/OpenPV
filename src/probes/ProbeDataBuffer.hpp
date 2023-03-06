#ifndef PROBEDATABUFFER_HPP_
#define PROBEDATABUFFER_HPP_

#include "ProbeData.hpp"
#include "utils/PVLog.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace PV {

template <typename T>
class ProbeDataBuffer {
  public:
   typedef typename std::vector<ProbeData<T>>::size_type size_type;
   ProbeDataBuffer(size_type batchWidth);
   ProbeDataBuffer();
   ~ProbeDataBuffer() {}

   static unsigned int calcPackedSize(unsigned int bufferSize, unsigned int batchWidth);

   void clear();

   std::vector<char> pack() const;
   static ProbeDataBuffer<T> unpack(std::vector<char> const &packedData);

   size_type size() const { return mBuffer.size(); }

   void store(ProbeData<T> const &newData);

   std::vector<ProbeData<T>> &getBuffer() { return mBuffer; }
   std::vector<ProbeData<T>> const &getBuffer() const { return mBuffer; }

   size_type getBatchWidth() const { return mBatchWidth; }

   ProbeData<T> &getData(int a) { return mBuffer.at(a); }
   ProbeData<T> const &getData(int a) const { return mBuffer.at(a); }

   double getTimestamp(int a) const { return mBuffer.at(a).getTimestamp(); }
   typename std::vector<T>::size_type getBatchWidth() { return mBatchWidth; }

   T &getValue(int a, int b) { return mBuffer.at(a).getValue(b); }
   T const &getValue(int a, int b) const { return mBuffer.at(a).getValue(b); }

  private:
   std::vector<ProbeData<T>> mBuffer;
   size_type mBatchWidth;
};

template <typename T>
ProbeDataBuffer<T>::ProbeDataBuffer(size_type batchWidth) {
   mBatchWidth = batchWidth;
}

template <typename T>
ProbeDataBuffer<T>::ProbeDataBuffer() {
   mBatchWidth = 0U;
}

template <typename T>
unsigned int ProbeDataBuffer<T>::calcPackedSize(unsigned int bufferSize, unsigned int batchWidth) {
   auto packedSizePerElement = ProbeData<T>::calcPackedSize(batchWidth);
   unsigned int headerSize   = 2U * static_cast<unsigned int>(sizeof(unsigned int));
   unsigned int dataSize     = static_cast<unsigned int>(packedSizePerElement) * bufferSize;
   unsigned int totalSize    = headerSize + dataSize;
   return static_cast<unsigned int>(totalSize);
}

template <typename T>
void ProbeDataBuffer<T>::clear() {
   mBuffer.clear();
}

template <typename T>
std::vector<char> ProbeDataBuffer<T>::pack() const {
   unsigned int bufferSize = size();
   auto packedLength       = calcPackedSize(bufferSize, mBatchWidth);
   std::vector<char> result(packedLength);
   memcpy(&result.at(0), &bufferSize, sizeof(unsigned int));
   auto batchWidth = static_cast<unsigned int>(mBatchWidth);
   memcpy(&result.at(sizeof(unsigned int)), &batchWidth, sizeof(unsigned int));
   for (unsigned int k = 0; k < bufferSize; ++k) {
      std::vector<char> packedElement = mBuffer[k].pack();
      unsigned int targetPosition     = calcPackedSize(k, mBatchWidth);
      std::copy(packedElement.begin(), packedElement.end(), &result.at(targetPosition));
   }
   return result;
}

template <typename T>
void ProbeDataBuffer<T>::store(ProbeData<T> const &newData) {
   if (!mBatchWidth) {
      mBatchWidth = newData.size();
   }
   else {
      FatalIf(
            mBatchWidth != newData.size(),
            "ProbeDataBuffer::store() argument has batchwidth %u but buffer batch width is %u\n",
            (unsigned int)newData.size(),
            (unsigned int)mBatchWidth);
   }
   double timestamp                              = newData.getTimestamp();
   typename std::vector<T>::size_type batchWidth = newData.size();
   mBuffer.emplace_back(timestamp, batchWidth);
   auto *dest      = &mBuffer.back().getValue(0);
   auto const *src = &newData.getValue(0);
   auto copySize   = batchWidth * sizeof(T);
   std::memcpy(dest, src, copySize);
}

template <typename T>
ProbeDataBuffer<T> ProbeDataBuffer<T>::unpack(std::vector<char> const &packedData) {
   if (packedData.size() < 2U * sizeof(unsigned int)) {
      throw std::runtime_error("ProbeData::unpack() argument is too short.");
   }
   unsigned int headerData[2]; // headerData[0] is bufferSize; headerData[1] is batchWidth
   memcpy(headerData, packedData.data(), 2U * sizeof(unsigned int));
   ProbeDataBuffer<T> result(headerData[1]);
   unsigned int sizePerElement = ProbeData<T>::calcPackedSize(headerData[1]);
   unsigned int correctSize    = 2U * sizeof(unsigned int) + headerData[0] * sizePerElement;
   if (static_cast<int>(packedData.size()) != correctSize) {
      std::string errorMessage(
            "ProbeDataBuffer::unpack() argument has #1 bytes but header indicates "
            "it should have #2 bytes.");
      errorMessage.replace(errorMessage.find("#1"), 2, std::to_string(packedData.size()));
      errorMessage.replace(errorMessage.find("#2"), 2, std::to_string(correctSize));
      throw std::runtime_error(errorMessage);
   }

   std::vector<char> elementPacked(sizePerElement);
   for (unsigned int k = 0; k < headerData[0]; ++k) {
      unsigned int position = 2U * sizeof(unsigned int) + k * sizePerElement;
      memcpy(&elementPacked.at(0), &packedData[position], sizePerElement);
      ProbeData<T> element = ProbeData<T>::unpack(elementPacked);
      result.store(element);
   }
   return result;
}

} // end namespace PV

#endif // PROBEDATABUFFER_HPP_
