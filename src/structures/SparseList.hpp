#ifndef __SPARSELIST_HPP__
#define __SPARSELIST_HPP__

#include "Buffer.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using std::vector;

namespace PV {

template <typename T>
class SparseList {
  public:
   typedef struct {
      uint32_t index;
      T value;
   } Entry;

   SparseList() {}
   SparseList(const Buffer<T> &source, T zeroVal) { fromBuffer(source, zeroVal); }

   void fromBuffer(const Buffer<T> &source, T zeroVal) {
      mList.clear();
      uint32_t index = 0;
      for (int y = 0; y < source.getHeight(); ++y) {
         for (int x = 0; x < source.getWidth(); ++x) {
            for (int f = 0; f < source.getFeatures(); ++f) {
               T val = source.at(x, y, f);
               if (val != zeroVal) {
                  mList.push_back({index, val});
               }
               index++;
            }
         }
      }
   }

   void toBuffer(Buffer<T> &dest, T zeroVal) {
      int numElements = dest.getTotalElements();
      vector<T> newData(numElements, zeroVal);
      for (auto entry : mList) {
         pvErrorIf(
               entry.index > numElements,
               "Buffer is not large enough to hold index %d / %d\n",
               entry.index,
               numElements);
         newData.at(entry.index) = entry.value;
      }
      dest.set(newData, dest.getWidth(), dest.getHeight(), dest.getFeatures());
   }

   void addEntry(Entry entry) { mList.push_back(entry); }

   void appendToList(SparseList<T> &dest) {
      for (auto entry : mList) {
         dest.addEntry(entry);
      }
   }

   void set(const vector<Entry> &values) { mList = values; }

   vector<Entry> getContents() { return mList; }

  private:
   vector<Entry> mList;
};
}

#endif
