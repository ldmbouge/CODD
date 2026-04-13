#pragma once

#include <cassert>
#include <initializer_list>
#include <type_traits>

#include "BitUtils.hpp"
#include "FunQual.hpp"
#include "HashUtils.hpp"
#include "MathUtils.hpp"
#include "Types.hpp"

namespace gfl {
template<i32 NumWords = 1>
class BitSet {
public:
  using WordType = u64;
  static constexpr i32 WordBitSize = numeric_limits<WordType>::digits;

private:
  static_assert(NumWords > 0);
  static_assert(std::is_integral_v<WordType>);
  static_assert(std::is_unsigned_v<WordType>);

  WordType _words[NumWords];

public:
  GFL_HOST_DEVICE
  BitSet() noexcept { clear(); }

  GFL_HOST_DEVICE
  BitSet(i32 const val) noexcept : BitSet() { insert(val); }

  GFL_HOST_DEVICE
  BitSet(i32 const min, i32 max) noexcept : BitSet() {
    assert(min >= 0);
    assert(max >= 0);
    assert(min <= max);
    assert(max < capacity());

    i32 const minWordIdx = min / WordBitSize;
    i32 const minBitIdx = min % WordBitSize;
    i32 const maxWordIdx = max / WordBitSize;
    i32 const maxBitIdx = max % WordBitSize;

    constexpr WordType fullWord = numeric_limits<WordType>::max();
    auto const minWordMask = suffixMaskInclusive<WordType>(minBitIdx);
    auto const maxWordMask = prefixMaskInclusive<WordType>(maxBitIdx);

    if (minWordIdx == maxWordIdx) {
      _words[minWordIdx] = minWordMask & maxWordMask;
    } else if constexpr (NumWords > 1) {
      _words[minWordIdx] = minWordMask;
      for (auto i = minWordIdx + 1; i < maxWordIdx; ++i) {
        _words[i] = fullWord;
      }
      _words[maxWordIdx] = maxWordMask;
    }
  }

  BitSet(std::initializer_list<i32> const vals) noexcept : BitSet()
  {
    for (i32 const & v : vals) {
      insert(v);
    }
  }

  GFL_HOST_DEVICE static constexpr
  i32 numWords(i32 const max) noexcept {
    return ceilDiv<i32>(max + 1, WordBitSize);
  }

  GFL_HOST_DEVICE
  void clear() noexcept {
    for (auto i = 0; i < NumWords; ++i) {
      _words[i] = 0;
    }
  }

  GFL_HOST_DEVICE
  void insert(i32 val) noexcept {
    assert(0 <= val);
    assert(val < capacity());

    i32 const wordIdx = val / WordBitSize;
    i32 const bitIdx = val % WordBitSize;
    _words[wordIdx] |= mask<WordType>(bitIdx);
  }

  GFL_HOST_DEVICE static constexpr
  i32 capacity() noexcept { return WordBitSize * NumWords; }

  GFL_HOST_DEVICE
  i32 size() const noexcept {
    i32 count = 0;
    for (auto i = 0; i < NumWords; ++i) {
      count += popcount(_words[i]);
    }
    return count;
  }

  GFL_HOST_DEVICE
  tuple<i32, i32, i32> summary() const noexcept {
    i32 smallest = numeric_limits<i32>::max();
    i32 largest = numeric_limits<i32>::min();
    i32 count = 0;

    for (auto i = 0; i < NumWords; ++i) {
      i32 const wBegin = i * WordBitSize;
      i32 const s = _words[i] != 0 ? wBegin + lsb(_words[i]) : numeric_limits<i32>::max();
      i32 const l = _words[i] != 0 ? wBegin + msb(_words[i]) : numeric_limits<i32>::min();
      i32 const c = _words[i] != 0 ? popcount(_words[i]) : 0;
      smallest = gfl::min<i32>(smallest, s);
      largest = gfl::max<i32>(largest, l);
      count += c;
    }
    return {smallest, largest, count};
  }

  GFL_HOST_DEVICE
  bool empty() const noexcept {
    for (auto i = 0; i < NumWords; ++i) {
      if (_words[i] != 0) return false;
    }
    return true;
  }

  GFL_HOST_DEVICE
  i32 rank(i32 val) const noexcept {
    i32 count = 0;
    i32 const wordIdx = val / WordBitSize;
    i32 const bitIdx = val % WordBitSize;

    for (auto i = 0; i < wordIdx; ++i) {
      count += popcount(_words[i]);
    }
    if (wordIdx < NumWords and bitIdx > 0) {
      auto const mask = prefixMaskInclusive<WordType>(bitIdx - 1);
      count += popcount(_words[wordIdx] & mask);
    }
    return count;
  }

  GFL_HOST_DEVICE
  void remove(i32 val) noexcept {
    assert(0 <= val);
    assert(val < capacity());

    i32 const wordIdx = val / WordBitSize;
    i32 const bitIdx = val % WordBitSize;
    _words[wordIdx] &= ~mask<WordType>(bitIdx);
  }

  GFL_HOST_DEVICE
  bool contains(i32 val) const noexcept {
    if (0 <= val and val < capacity()) {
      i32 const wordIdx = val / WordBitSize;
      i32 const bitIdx = val % WordBitSize;
      return test(_words[wordIdx], bitIdx);
    }
    return false;
  }

  GFL_HOST_DEVICE
  i32 smallest() const noexcept {
    i32 smallest = numeric_limits<i32>::max();
    for (auto i = 0; i < NumWords; ++i) {
      i32 const wBegin = i * WordBitSize;
      i32 const s = _words[i] != 0 ? wBegin + lsb(_words[i]) : numeric_limits<i32>::max();
      smallest = min<int>(smallest, s);
    }
    return smallest;
  }

  GFL_HOST_DEVICE
  i32 largest() const noexcept {
    i32 largest = numeric_limits<i32>::min();
    for (auto i = 0; i < NumWords; ++i) {
      i32 const wBegin = i * WordBitSize;
      i32 const l = _words[i] != 0 ? wBegin + msb(_words[i]) : numeric_limits<i32>::min();
      largest = max<int>(largest, l);
    }
    return largest;
  }

  GFL_HOST_DEVICE
  u64 hash() const noexcept {
    u64 seed = 0;
    for (auto i = 0; i < NumWords; ++i) {
      hashCombine(seed, _words[i]);
    }
    return seed;
  }

  GFL_HOST_DEVICE
  void print(char const * fmt = "%d") const noexcept {
    bool comma = false;
    for (auto const v : this) {
      printf(comma ? "," : "");
      printf(fmt, v);
      comma = true;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, BitSet const& set) {
    bool comma = false;
    for (auto const v : set) {
      os << (comma ? "," : "");
      os << v;
      comma = true;
    }
    return os;
  }

  GFL_HOST_DEVICE
  bool isEqual(BitSet const & other) const noexcept {
    for (auto i = 0; i < NumWords; ++i) {
      if (_words[i] != other._words[i]) return false;
    }
    return true;
  }

  GFL_HOST_DEVICE
  BitSet & unionWith(BitSet const & other) noexcept {
    for (auto wIdx = 0; wIdx < NumWords; ++wIdx) {
      _words[wIdx] |= other._words[wIdx];
    }
    return *this;
  }

  GFL_HOST_DEVICE
  BitSet & interWith(BitSet const & other) noexcept {
    for (auto wIdx = 0; wIdx < NumWords; ++wIdx) {
      _words[wIdx] &= other._words[wIdx];
    }
    return *this;
  }

  GFL_HOST_DEVICE
  BitSet & diffWith(BitSet const & other) noexcept {
    for (auto wIdx = 0; wIdx < NumWords; ++wIdx) {
      _words[wIdx] &= ~other._words[wIdx];
    }
    return *this;
  }

  GFL_HOST_DEVICE
  BitSet & complement() noexcept {
    for (auto wIdx = 0; wIdx < NumWords; ++wIdx) {
      _words[wIdx] = ~_words[wIdx];
    }
    return *this;
  }

  GFL_HOST_DEVICE friend
  bool operator==(BitSet const & a, BitSet const & b) noexcept { return a.isEqual(b); }

  GFL_HOST_DEVICE friend
  bool operator!=(BitSet const & a, BitSet const & b) noexcept { return not(a == b); }

  GFL_HOST_DEVICE friend
  BitSet operator|(BitSet a, BitSet const & b) noexcept {
    a.unionWith(b);
    return a;
  }

  GFL_HOST_DEVICE friend
  BitSet operator&(BitSet a, BitSet const & b) noexcept {
    a.interWith(b);
    return a;
  }

  GFL_HOST_DEVICE friend
  BitSet operator-(BitSet a, BitSet const & b) noexcept {
    a.diffWith(b);
    return a;
  }

  GFL_HOST_DEVICE friend
  BitSet operator~(BitSet s) noexcept {
    s.complement();
    return s;
  }

  GFL_HOST_DEVICE friend
  bool operator<=(BitSet const & a, BitSet const & b) noexcept {
    i32 count = 0;
    for (auto i = 0; i < NumWords; ++i) {
      count += (a._words[i] & b._words[i]) == a._words[i];
    }
    return count == NumWords;
  }

  class Iterator {
    WordType const * _words;
    i32 _idx;

    GFL_HOST_DEVICE
    void advance() {
      while (_idx < WordBitSize * NumWords) {
        i32 const wIdx = _idx / WordBitSize;
        i32 const bitIdx = _idx % WordBitSize;
        auto const remaining = _words[wIdx] >> bitIdx;
        if (remaining != 0) {
          _idx += lsb(remaining);
          return;
        }
        _idx = (wIdx + 1) * WordBitSize;
      }
    }

  public:
    GFL_HOST_DEVICE
    Iterator(WordType const * words, i32 const idx) :
      _words(words),
      _idx(idx)
    {
      advance();
    }

    GFL_HOST_DEVICE
    i32 operator*() const { return _idx; }

    GFL_HOST_DEVICE
    Iterator & operator++() {
      ++_idx;
      advance();
      return *this;
    }

    GFL_HOST_DEVICE
    bool operator==(Iterator const & o) const { return _idx == o._idx; }

    GFL_HOST_DEVICE
    bool operator!=(Iterator const & o) const { return _idx != o._idx; }
  };

  GFL_HOST_DEVICE
  Iterator begin() const noexcept { return Iterator(_words, 0); }

  GFL_HOST_DEVICE
  Iterator end() const noexcept { return Iterator(_words, WordBitSize * NumWords); }
};
}
