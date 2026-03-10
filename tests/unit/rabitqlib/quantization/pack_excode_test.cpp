#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "rabitqlib/quantization/pack_excode.hpp"

namespace {

void fill_random(std::vector<uint8_t>& data, uint8_t max_value, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(max_value));
    for (auto& v : data) {
        v = static_cast<uint8_t>(dist(rng));
    }
}

void pack_3bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t i = 0; i < 16; ++i) {
            uint8_t c = static_cast<uint8_t>(o_raw[i] & 0x3U);
            c |= static_cast<uint8_t>((o_raw[i + 16] & 0x3U) << 2);
            c |= static_cast<uint8_t>((o_raw[i + 32] & 0x3U) << 4);
            c |= static_cast<uint8_t>((o_raw[i + 48] & 0x3U) << 6);
            o_compact[i] = c;
        }
        o_compact += 16;

        uint64_t top_bit = 0;
        for (size_t idx = 0; idx < 64; ++idx) {
            const uint64_t bit = static_cast<uint64_t>((o_raw[idx] >> 2) & 0x1U);
            const size_t pos = ((idx & 7UL) << 3U) | (idx >> 3U);
            top_bit |= (bit << pos);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));
        o_raw += 64;
        o_compact += 8;
    }
}

void pack_1bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t j = 0; j < dim; j += 16) {
        uint16_t code = 0;
        for (size_t i = 0; i < 16; ++i) {
            code |= static_cast<uint16_t>(o_raw[i] & 0x1U) << i;
        }
        std::memcpy(o_compact, &code, sizeof(uint16_t));
        o_raw += 16;
        o_compact += 2;
    }
}

void pack_2bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t i = 0; i < 16; ++i) {
            uint8_t c = static_cast<uint8_t>(o_raw[i] & 0x3U);
            c |= static_cast<uint8_t>((o_raw[i + 16] & 0x3U) << 2);
            c |= static_cast<uint8_t>((o_raw[i + 32] & 0x3U) << 4);
            c |= static_cast<uint8_t>((o_raw[i + 48] & 0x3U) << 6);
            o_compact[i] = c;
        }
        o_raw += 64;
        o_compact += 16;
    }
}

void pack_4bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t j = 0; j < dim; j += 16) {
        for (size_t i = 0; i < 8; ++i) {
            o_compact[i] = static_cast<uint8_t>((o_raw[i] & 0x0FU) |
                                                ((o_raw[i + 8] & 0x0FU) << 4));
        }
        o_raw += 16;
        o_compact += 8;
    }
}

void pack_5bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t i = 0; i < 16; ++i) {
            o_compact[i] = static_cast<uint8_t>((o_raw[i] & 0x0FU) |
                                                ((o_raw[i + 16] & 0x0FU) << 4));
            o_compact[i + 16] = static_cast<uint8_t>((o_raw[i + 32] & 0x0FU) |
                                                     ((o_raw[i + 48] & 0x0FU) << 4));
        }
        o_compact += 32;
        uint64_t top_bit = 0;
        for (size_t idx = 0; idx < 64; ++idx) {
            const uint64_t bit = static_cast<uint64_t>((o_raw[idx] >> 4) & 0x1U);
            const size_t pos = ((idx & 7UL) << 3U) | (idx >> 3U);
            top_bit |= (bit << pos);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));
        o_raw += 64;
        o_compact += 8;
    }
}

void pack_6bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t v0 = static_cast<uint8_t>(o_raw[i] & 0x3FU);
            const uint8_t v1 = static_cast<uint8_t>(o_raw[i + 16] & 0x3FU);
            const uint8_t v2 = static_cast<uint8_t>(o_raw[i + 32] & 0x3FU);
            const uint8_t v3 = static_cast<uint8_t>(o_raw[i + 48] & 0x3FU);
            o_compact[i] = static_cast<uint8_t>(v0 | ((v3 & 0x03U) << 6));
            o_compact[i + 16] = static_cast<uint8_t>(v1 | (((v3 >> 2) & 0x03U) << 6));
            o_compact[i + 32] = static_cast<uint8_t>(v2 | (((v3 >> 4) & 0x03U) << 6));
        }
        o_compact += 48;
        o_raw += 64;
    }
}

void pack_7bit_ref(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t v0 = static_cast<uint8_t>(o_raw[i] & 0x3FU);
            const uint8_t v1 = static_cast<uint8_t>(o_raw[i + 16] & 0x3FU);
            const uint8_t v2 = static_cast<uint8_t>(o_raw[i + 32] & 0x3FU);
            const uint8_t v3 = static_cast<uint8_t>(o_raw[i + 48] & 0x3FU);
            o_compact[i] = static_cast<uint8_t>(v0 | ((v3 & 0x03U) << 6));
            o_compact[i + 16] = static_cast<uint8_t>(v1 | (((v3 >> 2) & 0x03U) << 6));
            o_compact[i + 32] = static_cast<uint8_t>(v2 | (((v3 >> 4) & 0x03U) << 6));
        }
        o_compact += 48;
        uint64_t top_bit = 0;
        for (size_t idx = 0; idx < 64; ++idx) {
            const uint64_t bit = static_cast<uint64_t>((o_raw[idx] >> 6) & 0x1U);
            const size_t pos = ((idx & 7UL) << 3U) | (idx >> 3U);
            top_bit |= (bit << pos);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));
        o_compact += 8;
        o_raw += 64;
    }
}

}  // namespace

TEST(ExcodePackingTest, Pack3BitMatchesReference) {
    std::mt19937 rng(42);
    const size_t dim = 128;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 7, rng);

    const size_t out_size = (dim / 64) * 24;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_3bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_3bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack1BitMatchesReference) {
    std::mt19937 rng(40);
    const size_t dim = 64;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 1, rng);

    const size_t out_size = dim / 8;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_1bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_1bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack2BitMatchesReference) {
    std::mt19937 rng(41);
    const size_t dim = 128;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 3, rng);

    const size_t out_size = (dim / 64) * 16;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_2bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_2bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack4BitMatchesReference) {
    std::mt19937 rng(43);
    const size_t dim = 32;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 15, rng);

    const size_t out_size = dim / 2;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_4bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_4bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack5BitMatchesReference) {
    std::mt19937 rng(44);
    const size_t dim = 128;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 31, rng);

    const size_t out_size = (dim / 64) * 40;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_5bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_5bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack6BitMatchesReference) {
    std::mt19937 rng(45);
    const size_t dim = 128;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 63, rng);

    const size_t out_size = (dim / 64) * 48;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_6bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_6bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}

TEST(ExcodePackingTest, Pack7BitMatchesReference) {
    std::mt19937 rng(46);
    const size_t dim = 128;
    std::vector<uint8_t> raw(dim);
    fill_random(raw, 127, rng);

    const size_t out_size = (dim / 64) * 56;
    std::vector<uint8_t> ref(out_size, 0);
    std::vector<uint8_t> out(out_size, 0);

    pack_7bit_ref(raw.data(), ref.data(), dim);
    rabitqlib::quant::rabitq_impl::ex_bits::packing_7bit_excode(raw.data(), out.data(), dim);

    ASSERT_EQ(ref, out);
}
