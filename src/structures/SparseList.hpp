#ifndef SPARSELIST_HPP_
#define SPARSELIST_HPP_

#include "structures/Buffer.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <algorithm>
#include <cinttypes>
#include <vector>

using std::vector;

namespace PV {

template <typename T>
class SparseList {
  public:
   struct Entry {
      uint32_t index;
      T value;
   };

   SparseList() {}
   SparseList(const Buffer<T> &source, T zeroVal) { fromBuffer(source, zeroVal); }
   SparseList(int width, int height, int features) { reset(width, height, features); }

   void fromBuffer(const Buffer<T> &source, T zeroVal) {
      reset(source.getWidth(), source.getHeight(), source.getFeatures());
      uint32_t index = 0;
      for (int y = 0; y < source.getHeight(); ++y) {
         for (int x = 0; x < source.getWidth(); ++x) {
            for (int f = 0; f < source.getFeatures(); ++f) {
               T val = source.at(x, y, f);
               if (val != zeroVal) {
                  mContents.push_back({index, val});
               }
               ++index;
            }
         }
      }
   }

   void reset(int width, int height, int features) {
      mWidth = width;
      mHeight = height;
      mFeatures = features;
      mContents.clear();
   }

   void toBuffer(Buffer<T> &dest, T zeroVal) {
      uint32_t numElements = dest.getTotalElements();
      vector<T> newData(numElements, zeroVal);
      for (auto entry : mContents) {
         FatalIf(
               entry.index > numElements,
               "Buffer is not large enough to hold index %" PRIu32 " / %" PRIu32 "\n",
               entry.index,
               numElements);
         newData.at(entry.index) = entry.value;
      }
      dest.set(newData, dest.getWidth(), dest.getHeight(), dest.getFeatures());
   }

   void addEntry(Entry const &entry) {
      auto pos = std::find_if(
            mContents.begin(), mContents.end(), [entry](Entry a) { return a.index > entry.index; });
      mContents.insert(pos, entry);
   }

   void addEntry(int index, T value) {
      Entry entry{static_cast<uint32_t>(index), value};
      addEntry(entry);
   }

   void crop(int newWidth, int newHeight, int leftCrop, int topCrop) {
      FatalIf(
            newWidth > getWidth(),
            "SparseList::crop called with existing width %d, new width %d\n",
            getWidth(),
            newWidth);
      FatalIf(
            newHeight > getHeight(),
            "SparseList::crop called with existing height %d, new height %d\n",
            getHeight(),
            newHeight);
      FatalIf(
            leftCrop > getWidth() - newWidth,
            "SparseList::crop called with leftCrop %d but size reduction %d\n",
            leftCrop,
            getWidth() - newWidth);
      FatalIf(
            topCrop > getHeight() - newHeight,
            "SparseList::crop called with topCrop %d but added margin %d\n",
            topCrop,
            getHeight() - newHeight);
      int numValues = static_cast<int>(mContents.size());
      vector<Entry> newContents;
      for (int n = 0; n < numValues; ++n) {
         int k = mContents[n].index;
         // Should use conversions.hpp routines for this, but nvcc gives undefined identifer errors.
         int kx = (k / getFeatures()) % getWidth() - leftCrop;
         int ky = k / (getWidth() * getFeatures()) % getHeight() - topCrop;
         int kf = k % getFeatures();
         bool xInBounds = kx >= 0 and kx < newWidth;
         bool yInBounds = ky >= 0 and ky < newHeight;
         if (xInBounds and yInBounds) {
            uint32_t newIndex = static_cast<uint32_t>(kf + getFeatures() * (kx + newWidth * ky));
            newContents.emplace_back(Entry{newIndex, mContents[n].value});
         }
      }
      set(newContents);
      mWidth = newWidth;
      mHeight = newHeight;
   }

   SparseList extract(int xStart, int yStart, int width, int height) const {
      FatalIf(
            xStart < 0,
            "SparseList::extract called with negative xStart value %d\n",
            xStart);
      FatalIf(
            yStart < 0,
            "SparseList::extract called with negative yStart value %d\n",
            yStart);
      FatalIf(
            xStart + width > getWidth(),
            "SparseList::extract called with xStart + width too large (%d+%d versus %d)\n",
            xStart, width, getWidth());
      FatalIf(
            yStart + height > getHeight(),
            "SparseList::extract called with yStart + height too large (%d+%d versus %d)\n",
            yStart, height, getHeight());
      SparseList result;
      for (auto &e : mContents) {
         int k = e.index;
         // Should use conversions.hpp routines for this, but nvcc gives undefined identifer errors.
         int kx = (k / getFeatures()) % getWidth() - xStart;
         int ky = k / (getWidth() * getFeatures()) % getHeight() - yStart;
         int kf = k % getFeatures();
         bool xInBounds = kx >= 0 and kx < width;
         bool yInBounds = ky >= 0 and ky < height;
         if (xInBounds and yInBounds) {
            uint32_t newIndex = static_cast<uint32_t>(kf + getFeatures() * (kx + width * ky));
            result.addEntry(newIndex, e.value);
         }
      }
      return result;
   }

   void grow(int newWidth, int newHeight, int leftPad, int topPad) {
      FatalIf(
            newWidth < getWidth(),
            "SparseList::grow called with existing width %d, new width %d\n",
            getWidth(),
            newWidth);
      FatalIf(
            newHeight < getHeight(),
            "SparseList::grow called with existing height %d, new height %d\n",
            getHeight(),
            newHeight);
      FatalIf(
            leftPad > newWidth - getWidth(),
            "SparseList::grow called with leftPad %d but added margin %d\n",
            leftPad,
            newWidth - getWidth());
      FatalIf(
            topPad > newHeight - getHeight(),
            "SparseList::grow called with topPad %d but added margin %d\n",
            topPad,
            newHeight - getHeight());
      int numValues = static_cast<int>(mContents.size());
      vector<Entry> newContents(numValues);
      for (int n = 0; n < numValues; ++n) {
         int k = mContents[n].index;
         // Should use conversions.hpp routines for this, but nvcc gives undefined identifer errors.
         int kx = (k / getFeatures()) % getWidth() + leftPad;
         pvAssert(kx >= 0 and kx < newWidth);
         int ky = k / (getWidth() * getFeatures()) % getHeight() + topPad;
         pvAssert(ky >= 0 and ky < newHeight);
         int kf = k % getFeatures();
         uint32_t newIndex = static_cast<uint32_t>(kf + getFeatures() * (kx + newWidth * ky));
         newContents[n].index = newIndex;
         newContents[n].value = mContents[n].value;
      }
      set(newContents);
      mWidth = newWidth;
      mHeight = newHeight;
   }

   void merge(SparseList const &newList) {
      FatalIf(
            newList.getWidth() != getWidth(),
            "SparseList::merge called with a list of different width (%d versus %d)\n",
            newList.getWidth(),
            getWidth());
      FatalIf(
            newList.getHeight() != getHeight(),
            "SparseList::merge called with a list of different height (%d versus %d)\n",
            newList.getHeight(),
            getHeight());
      FatalIf(
            newList.getFeatures() != getFeatures(),
            "SparseList::merge called with a list of different height (%d versus %d)\n",
            newList.getFeatures(),
            getFeatures());
      
      auto newContents = newList.getContents();
      vector<Entry> mergedContents;
      std::merge(
            mContents.begin(),
            mContents.end(),
            newContents.begin(),
            newContents.end(),
            std::back_inserter(mergedContents),
            [](Entry a, Entry b){return a.index < b.index;});

      auto testequal = [](Entry a, Entry b) {
         bool indexequals = a.index == b.index;
         bool valueequals = a.value == b.value;
         if (indexequals and !valueequals) {
            WarnLog() << "SparseList has two elements with same index but different values\n";
         }
         return indexequals and valueequals;
      };
      auto newEnd = std::unique(mergedContents.begin(), mergedContents.end(), testequal);
      mergedContents.erase(newEnd, mergedContents.end());
      mContents = mergedContents;
   }

   int getWidth() const { return mWidth; }
   int getHeight() const { return mHeight; }
   int getFeatures() const { return mFeatures; }

   void set(const vector<Entry> &values) { mContents = values; }

   vector<Entry> getContents() const { return mContents; }

  private:
   int mWidth = 0;
   int mHeight = 0;
   int mFeatures = 0;
   vector<Entry> mContents;
}; // class SparseList<T>

} // namespace PV

#endif // SPARSELIST_HPP_
