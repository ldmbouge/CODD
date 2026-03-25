#pragma once

#include <cassert>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "BitUtils.hpp"
#include "MathUtils.hpp"
#include "FunQual.hpp"
#include "Types.hpp"
#include "HashUtils.hpp"

namespace gfl
{
    template<i32 NumWords = 1>
    class BitSet
    {
        static_assert(NumWords > 0);

    public:
        using WordType = u64;
        static constexpr i32 WordBitSize = std::numeric_limits<WordType>::digits;

    private:

        static_assert(std::is_integral_v<WordType>);
        static_assert(std::is_unsigned_v<WordType>);
        WordType words_[NumWords]{};

    public:
        BitSet() noexcept = default;

        GFL_HOST_DEVICE BitSet(i32 val) noexcept;
        GFL_HOST_DEVICE BitSet(i32 min, i32 max) noexcept;
        GFL_HOST_DEVICE BitSet(std::initializer_list<i32> vals) noexcept;

        GFL_HOST_DEVICE static constexpr i32 num_words(i32 const maxValue) noexcept
        {
            return ceil<i32>(maxValue, WordBitSize);
        }

        GFL_HOST_DEVICE static constexpr i32 capacity() noexcept { return WordBitSize * NumWords; }

        GFL_HOST_DEVICE i32 size() const noexcept;
        GFL_HOST_DEVICE tuple<i32,i32,i32> summary() const noexcept;
        GFL_HOST_DEVICE static f32 iou(BitSet const& a, BitSet const& b) noexcept;
        GFL_HOST_DEVICE f32 iou(BitSet const& other ) const noexcept;
        GFL_HOST_DEVICE bool empty() const noexcept;

        GFL_HOST_DEVICE void clear() noexcept;

        GFL_HOST_DEVICE void insert(i32 val) noexcept;
        GFL_HOST_DEVICE void remove(i32 val) noexcept;
        GFL_HOST_DEVICE bool contains(i32 val) const noexcept;

        GFL_HOST_DEVICE bool isEqual(BitSet const& other) const noexcept;
        GFL_HOST_DEVICE BitSet& unionWith(BitSet const& other) noexcept;
        GFL_HOST_DEVICE BitSet& interWith(BitSet const& other) noexcept;
        GFL_HOST_DEVICE BitSet& diffWith(BitSet const& other) noexcept;
        GFL_HOST_DEVICE BitSet& complement() noexcept;

        GFL_HOST_DEVICE
        BitSet shiftLeft(i32 const k) const noexcept
        {
            assert(k > 0);

            BitSet result;
            i32 const wordShift = k / WordBitSize;
            i32 const bitShift  = k % WordBitSize;

            for (i32 i = 0; i < NumWords; ++i)
            {
                i32 const src = i - wordShift;
                // src contributes upper bits, src-1 contributes lower bits (overflow)
                WordType const upper = (src >= 0 and src < NumWords) ? words_[src] << bitShift : 0;
                WordType const lower = (bitShift > 0 and src - 1 >= 0 and src - 1 < NumWords) ? words_[src - 1] >> (WordBitSize - bitShift) : 0;
                result.words_[i] = upper | lower;
            }
            return result;
        }

        GFL_HOST_DEVICE
        BitSet shiftRight(i32 const k) const noexcept
        {
            assert(k > 0);

            BitSet result;
            i32 const wordShift = k / WordBitSize;
            i32 const bitShift  = k % WordBitSize;

            for (i32 i = 0; i < NumWords; ++i)
            {
                i32 const src = i + wordShift;
                // src contributes lower bits, src+1 contributes upper bits (overflow)
                WordType const lower = (src >= 0 and src < NumWords) ? words_[src] >> bitShift : 0;
                WordType const upper = (bitShift > 0 and src + 1 < NumWords) ? words_[src + 1] << (WordBitSize - bitShift) : 0;
                result.words_[i] = lower | upper;
            }
            return result;
        }

        GFL_HOST_DEVICE
        BitSet reversed() const noexcept
        {
            using namespace gfl;
            BitSet result;
            for (i32 i = 0; i < NumWords; ++i)
            {
                result.words_[i] = bitreverse(words_[NumWords - 1 - i]);
            }
            return result;
        }

        GFL_HOST_DEVICE
        BitSet(i32 const ofs, BitSet const & s) noexcept : words_{}
        {
             // Computes { ofs - v | v in s }
             // for (i32 i = 0; i < capacity(); ++i)
             // {
             //     if (s.contains(i)) insert(ofs - i);
             // }

            *this = s.reversed().shiftRight(capacity() - ofs - 1);
        }

        GFL_HOST_DEVICE
        friend bool operator==(BitSet const& a, BitSet const& b) noexcept { return a.isEqual(b); }

        GFL_HOST_DEVICE
        friend bool operator!=(BitSet const& a, BitSet const& b) noexcept { return !(a == b); }

        GFL_HOST_DEVICE
        friend BitSet operator|(BitSet a, BitSet const& b) noexcept { a.unionWith(b); return a; }

        GFL_HOST_DEVICE
        friend BitSet operator&(BitSet a, BitSet const& b) noexcept { a.interWith(b); return a; }

        GFL_HOST_DEVICE
        friend BitSet operator-(BitSet a, BitSet const& b) noexcept { a.diffWith(b); return a; }

        GFL_HOST_DEVICE
        friend BitSet operator~(BitSet a) noexcept { a.complement(); return a; }

        GFL_HOST_DEVICE
        friend BitSet operator-(i32 const l, BitSet const & s) noexcept { return BitSet(l, s);}

        GFL_HOST_DEVICE
        friend bool operator<=(BitSet const& a, BitSet const& b) noexcept
        {
            unsigned short nw = 0;
            for (auto i = 0; i < NumWords; i++)
                nw += (a.words_[i] & b.words_[i]) == a.words_[i];
            return nw == NumWords;
        }

        template <typename Pred>
        GFL_HOST_DEVICE friend
        BitSet filter(BitSet const & set, Pred const p) {
            BitSet out;
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    if (p(v))
                        out.insert(v);
                }
            }
            return out;
        }

        GFL_HOST_DEVICE
        void printAsInts(i32 const end = WordBitSize * NumWords) const noexcept;

        GFL_HOST_DEVICE
        void printAs01(i32 const end = WordBitSize * NumWords) const noexcept;

        GFL_HOST_DEVICE
        void print() const noexcept;

        GFL_HOST_DEVICE
        i32 smallest() const noexcept;

        GFL_HOST_DEVICE
        i32 largest() const noexcept;

        GFL_HOST_DEVICE
        u64 hash() const noexcept
        {
            u64 seed = 0;
            for (auto i = 0;i < NumWords; i++) hashCombine(seed, words_[i]);
            return seed;
        }

        template <typename Term>
        GFL_HOST_DEVICE friend
        tuple<i32,i32> argmin(BitSet const & set, Term const & t)
        {
            i32 min = numeric_limits<i32>::max();
            int elt;
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    auto const tv = t(v);
                    min = (min < tv) ? min : tv;
                    elt = (min < tv) ? elt : v;
                }
            }
            return {elt, min};
        }

        template <typename Filter, typename Term>
        GFL_HOST_DEVICE friend
        i32 min(BitSet const & set, Filter const & f, Term const & t)
        {
            i32 ttl = numeric_limits<i32>::max();
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    if (f(v))
                    {
                        auto const tv = t(v);
                        ttl = (ttl  < tv) ? ttl : tv;
                    }
                }
            }
            return ttl;
        }

        template <typename Term>
        GFL_HOST_DEVICE friend
        i32 min(BitSet const & set, Term const & t)
        {
            i32 ttl = numeric_limits<i32>::max();
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    auto const tv = t(v);
                    ttl = (ttl  < tv) ? ttl : tv;
                }
            }
            return ttl;
        }

        template <typename Term>
        GFL_HOST_DEVICE friend
        i32 max(BitSet const & set, Term const & t)
        {
            i32 ttl = numeric_limits<i32>::min();
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    auto const tv = t(v);
                    ttl = (ttl > tv) ? ttl : tv;
                }
            }
            return ttl;
        }

        template <typename Filter>
        GFL_HOST_DEVICE friend
        i32 count(BitSet const & set, Filter const & f) {
            i32 count = 0;
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    if (f(v))
                    {
                        count++;
                    }
                }
            }
            return count;
        }

        template <typename Pred>
        GFL_HOST_DEVICE friend
        bool any(BitSet const & set, Pred const p)
        {
            auto [minVal,maxVal,_] = set.summary();
            for (i32 v = minVal; v <= maxVal; ++v)
            {
                if (set.contains(v))
                {
                    if (p(v)) return true;
                }
            }
            return false;
        }
    };


    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>::BitSet(i32 const val) noexcept : words_{}
    {
        insert(val);
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>::BitSet(i32 const min, i32 const max) noexcept : words_{}
    {
        assert(min >= 0);
        assert(max >= 0);
        assert(min <= max);
        assert(max < capacity());

        i32 const minWordIdx = min / WordBitSize;
        i32 const minBitIdx  = min % WordBitSize;
        i32 const maxWordIdx = max / WordBitSize;
        i32 const maxBitIdx  = max % WordBitSize;

        constexpr WordType fullWord = numeric_limits<WordType>::max();
        WordType const minWordMask = suffixMaskInclusive<WordType>(minBitIdx);
        WordType const maxWordMask = prefixMaskInclusive<WordType>(maxBitIdx);

        if (minWordIdx == maxWordIdx)
        {
            words_[minWordIdx] = minWordMask & maxWordMask;
        }
        else
        {
            words_[minWordIdx] = minWordMask;
            words_[maxWordIdx] = maxWordMask;

            for (i32 wIdx = minWordIdx + 1; wIdx < maxWordIdx; ++wIdx)
            {
                words_[wIdx] = fullWord;
            }
        }
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>::BitSet(std::initializer_list<i32> const vals) noexcept : words_{}
    {
        for (i32 const& v : vals)
        {
            insert(v);
        }
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    i32 BitSet<NumWords>::size() const noexcept
    {
        i32 count = 0;
        for (i32 i = 0; i < NumWords; ++i)
        {
            count += popcount(words_[i]);
        }
        return count;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    bool BitSet<NumWords>::empty() const noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            if (words_[i] != 0) return false;
        }
        return true;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    void BitSet<NumWords>::clear() noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            words_[i] = 0;
        }
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    void BitSet<NumWords>::insert(i32 const val) noexcept
    {
        using namespace gfl;

        assert(0 <= val);
        assert(val < capacity());

        i32 const wordIdx = val / WordBitSize;
        i32 const bitIdx  = val % WordBitSize;

        words_[wordIdx] |= mask<WordType>(bitIdx);
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    void BitSet<NumWords>::remove(i32 const val) noexcept
    {
        assert(0 <= val);
        assert(val < capacity());

        i32 const wordIdx = val / WordBitSize;
        i32 const bitIdx  = val % WordBitSize;

        words_[wordIdx] &= ~mask<WordType>(bitIdx);
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    bool BitSet<NumWords>::contains(i32 const val) const noexcept
    {
        if (0 <= val and val < capacity())
        {
            i32 const wordIdx = val / WordBitSize;
            i32 const bitIdx  = val % WordBitSize;
            return gfl::test(words_[wordIdx], bitIdx);
        }
        else return false;
    }

    template <i32 NumWords>
    GFL_HOST_DEVICE
    bool BitSet<NumWords>::isEqual(BitSet const& other) const noexcept
    {
        using namespace gfl;
        for (int i = 0; i < NumWords; ++i) if (words_[i] != other.words_[i]) return false;
        return true;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>& BitSet<NumWords>::unionWith(BitSet const& other) noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            words_[i] |= other.words_[i];
        }
        return *this;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>& BitSet<NumWords>::interWith(BitSet const& other) noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            words_[i] &= other.words_[i];
        }
        return *this;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>& BitSet<NumWords>::diffWith(BitSet const& other) noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            words_[i] &= ~other.words_[i];
        }
        return *this;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    BitSet<NumWords>& BitSet<NumWords>::complement() noexcept
    {
        for (i32 i = 0; i < NumWords; ++i)
        {
            words_[i] = ~words_[i];
        }
        return *this;
    }

    template<i32 NumWords>
    GFL_HOST
    void BitSet<NumWords>::printAsInts(i32 const end) const noexcept
    {
        bool comma = false;
        for(i32 val = 0; val < end; ++val)
        {
            if (contains(val))
            {
                printf(comma ? "," : "");
                printf("%d", val);
                comma = true;
            }
        }
    }

    template<i32 NumWords>
   GFL_HOST_DEVICE
   void BitSet<NumWords>::printAs01(i32 const end) const noexcept
    {
        bool comma = false;
        for (i32 val = 0; val < end; ++val)
        {
            printf(comma ? "," : "");
            printf(contains(val) ? "1" : "0");
            comma = true;
        }
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    void BitSet<NumWords>::print() const noexcept
    {
        for (i32 wIdx = 0; wIdx < NumWords; ++wIdx)
        {
            gfl::printBits(words_[wIdx]);
        }
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    i32 BitSet<NumWords>::smallest() const noexcept
    {
        using namespace gfl;

        i32 smallest = numeric_limits<i32>::max();
        for (i32 wIdx = 0; wIdx < NumWords; ++wIdx)
        {
            i32 const wBegin = wIdx * sizeof(unsigned long long) * 8;
            i32 const sWord = words_[wIdx] != 0 ? wBegin + lsb(words_[wIdx]) : numeric_limits<i32>::max();
            smallest = min<int>(smallest, sWord);
        }
        return smallest;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    i32 BitSet<NumWords>::largest() const noexcept
    {
        using namespace gfl;

        i32 largest = numeric_limits<i32>::min();
        for (i32 wIdx = 0; wIdx < NumWords; ++wIdx)
        {
            i32 const wBegin = wIdx * sizeof(unsigned long long) * 8;
            i32 const lWord = words_[wIdx] != 0 ? wBegin + msb(words_[wIdx]) : numeric_limits<i32>::min();
            largest = max<int>(largest, lWord);
        }
        return largest;
    }


    template<i32 NumWords>
    GFL_HOST_DEVICE
    tuple<int,int,int> BitSet<NumWords>::summary() const noexcept
    {
        using namespace gfl;

        i32 smallest = numeric_limits<i32>::max();
        i32 largest = numeric_limits<i32>::min();
        i32 count = 0;
        for (i32 wIdx = 0; wIdx < NumWords; ++wIdx)
        {
            i32 const wBegin = wIdx * sizeof(unsigned long long) * 8;
            i32 const sWord = words_[wIdx] != 0 ? wBegin + lsb(words_[wIdx]) : numeric_limits<i32>::max();
            i32 const lWord = words_[wIdx] != 0 ? wBegin + msb(words_[wIdx]) : numeric_limits<i32>::min();
            i32 const cWord = words_[wIdx] != 0 ? popcount(words_[wIdx]) : 0;
            smallest = gfl::min<i32>(smallest, sWord);
            largest = gfl::max<i32>(largest, lWord);
            count += cWord;
        }
        return {smallest,largest,count};
    }


    // Intersection Over Union
    // Reference: https://en.wikipedia.org/wiki/Jaccard_index
    template<i32 NumWords>
    GFL_HOST_DEVICE
    f32 BitSet<NumWords>::iou(BitSet const & a, BitSet const & b) noexcept
    {
        using namespace gfl;

        f32 intersectionSize = 0.0;
        f32 unionSize = 0.0;
        for (int wIdx = 0; wIdx < NumWords; wIdx += 1)
        {
            intersectionSize += popcount(a.words_[wIdx] & b.words_[wIdx]);
            unionSize += popcount(a.words_[wIdx] | b.words_[wIdx]);
        }
        return unionSize > 0.0 ? intersectionSize / unionSize : 0.0;
    }

    template<i32 NumWords>
    GFL_HOST_DEVICE
    f32 BitSet<NumWords>::iou(BitSet const & other) const noexcept
    {
        return iou(*this,other);
    }
}