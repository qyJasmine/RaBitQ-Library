#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf {
enum : uint32_t {
    kIvfPageTypeMeta = 1,
    kIvfPageTypeRotator = 2,
    kIvfPageTypeCentroids = 3,
    kIvfPageTypeDirectory = 4,
    kIvfPageTypeClusterData = 5,
};

static constexpr char kIvfMagicV2[8] = {'R', 'B', 'Q', 'I', 'V', 'F', '2', '\0'};

struct IvfPageHeaderV2 {
    uint32_t page_type = 0;
    uint16_t header_bytes = 0;
    uint16_t flags = 0;
    uint32_t page_id = 0;
    uint32_t next_page_id = std::numeric_limits<uint32_t>::max();
    uint32_t payload_bytes = 0;
    uint32_t crc32 = 0;
};

struct IvfMetaV2 {
    char magic[8] = {};
    uint32_t format_version = 2;
    uint32_t page_size = 0;
    uint64_t num_points = 0;
    uint32_t dim = 0;
    uint32_t padded_dim = 0;
    uint32_t num_clusters = 0;
    uint32_t ex_bits = 0;
    uint32_t metric_type = 0;
    uint32_t rotator_type = 0;
    uint32_t init_type = 0;
    uint32_t reserved0 = 0;
    uint32_t rotator_first_page = 0;
    uint32_t rotator_page_count = 0;
    uint32_t centroid_first_page = 0;
    uint32_t centroid_page_count = 0;
    uint32_t dir_first_page = 0;
    uint32_t dir_page_count = 0;
    uint32_t cluster_first_page = 0;
    uint32_t cluster_page_count = 0;
    uint64_t build_timestamp_unix = 0;
    uint64_t reserved1[4] = {};
};

struct IvfClusterDirEntryV2 {
    uint32_t cluster_id = 0;
    uint32_t num_points = 0;
    uint32_t num_batches = 0;
    uint32_t flags = 0;
    uint32_t first_page_id = 0;
    uint32_t last_page_id = 0;
    uint32_t first_batch_ordinal = 0;
    uint32_t reserved0 = 0;
    uint64_t cluster_bytes = 0;
    uint64_t batch_bytes = 0;
    uint64_t ex_bytes = 0;
    uint64_t ids_bytes = 0;
};

static_assert(sizeof(IvfPageHeaderV2) == 24, "IvfPageHeaderV2 size mismatch");
static_assert(sizeof(IvfClusterDirEntryV2) == 64, "IvfClusterDirEntryV2 size mismatch");

class IVF {
   private:
    enum class StorageMode : uint8_t { LegacyInMemory = 0, PagedV2 = 1 };
    static constexpr uint32_t kFormatV2 = 2;
    static constexpr size_t kPageSizeV2 = 8192;
    static constexpr size_t kPagePayloadV2 = kPageSizeV2 - sizeof(IvfPageHeaderV2);
    static constexpr size_t kDefaultPageCacheCapacityBytesV2 = 64UL * 1024UL * 1024UL;

    Initializer* initer_ = nullptr;      // initializer for find candidate cluster
    char* batch_data_ = nullptr;         // 1-bit code and factors
    char* ex_data_ = nullptr;            // code for remaining bits
    PID* ids_ = nullptr;                 // PID of vectors (orgnized by clusters)
    size_t num_;                         // num of data points
    size_t dim_;                         // dimension of data points
    size_t padded_dim_;                  // dimension after padding,
    size_t num_cluster_;                 // num of centroids (clusters)
    size_t ex_bits_;                     // total bits = ex_bits_ + 1
    RotatorType type_;                   // type of rotator
    Rotator<float>* rotator_ = nullptr;  // Data Rotator
    std::vector<Cluster> cluster_lst_;   // List of clusters in ivf
    MetricType metric_type_ = rabitqlib::METRIC_L2;  // metric type
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;
    StorageMode storage_mode_ = StorageMode::LegacyInMemory;
    std::string loaded_index_path_;
    std::vector<IvfClusterDirEntryV2> cluster_dir_v2_;
    int mapped_fd_ = -1;
    const char* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    struct CachedPageV2 {
        uint32_t page_id = 0;
        uint32_t page_type = 0;
        uint32_t next_page_id = std::numeric_limits<uint32_t>::max();
        uint32_t payload_bytes = 0;
        size_t pin_count = 0;
        std::vector<char> payload;
    };
    class PinnedPage {
       public:
        PinnedPage() = default;
        ~PinnedPage();
        PinnedPage(const PinnedPage&) = delete;
        PinnedPage& operator=(const PinnedPage&) = delete;
        PinnedPage(PinnedPage&&) noexcept;
        PinnedPage& operator=(PinnedPage&&) noexcept;
        [[nodiscard]] const char* payload() const { return payload_; }
        [[nodiscard]] size_t payload_bytes() const { return payload_bytes_; }
        [[nodiscard]] uint32_t next_page_id() const { return next_page_id_; }

       private:
        friend class IVF;
        const IVF* owner_ = nullptr;
        uint32_t page_id_ = std::numeric_limits<uint32_t>::max();
        uint32_t next_page_id_ = std::numeric_limits<uint32_t>::max();
        const char* payload_ = nullptr;
        size_t payload_bytes_ = 0;
        PinnedPage(
            const IVF* owner,
            uint32_t page_id,
            uint32_t next_page_id,
            const char* payload,
            size_t payload_bytes
        )
            : owner_(owner)
            , page_id_(page_id)
            , next_page_id_(next_page_id)
            , payload_(payload)
            , payload_bytes_(payload_bytes) {}
        void release();
    };
    mutable std::mutex page_cache_mu_;
    mutable std::list<CachedPageV2> page_cache_lru_;
    mutable std::unordered_map<uint32_t, std::list<CachedPageV2>::iterator> page_cache_index_;
    mutable size_t page_cache_bytes_ = 0;
    size_t page_cache_capacity_bytes_ = kDefaultPageCacheCapacityBytesV2;

    void quantize_cluster(
        Cluster&,
        const std::vector<PID>&,
        const float*,
        const float*,
        float*,
        const quant::RabitqConfig&
    );

    [[nodiscard]] size_t ids_bytes() const { return sizeof(PID) * num_; }

    // get num of bytes used for 1-bit code and corresponding factors
    [[nodiscard]] size_t batch_data_bytes(const std::vector<size_t>& cluster_sizes) const {
        assert(cluster_sizes.size() == num_cluster_);  // num of clusters
        size_t total_blocks = 0;
        for (auto size : cluster_sizes) {
            total_blocks += div_round_up(size, fastscan::kBatchSize);
        }
        return total_blocks * BatchDataMap<float>::data_bytes(padded_dim_);
    }

    [[nodiscard]] size_t ex_data_bytes() const {
        return ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * num_;
    }

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory() {
        ::delete initer_;
        initer_ = nullptr;
        std::free(batch_data_);
        batch_data_ = nullptr;
        std::free(ex_data_);
        ex_data_ = nullptr;
        std::free(ids_);
        ids_ = nullptr;
        cluster_lst_.clear();
        cluster_dir_v2_.clear();
        close_mapped_file();
        storage_mode_ = StorageMode::LegacyInMemory;
    }

    void search_cluster(
        const Cluster&, const SplitBatchQuery<float>&, buffer::SearchBuffer<float>&, bool
    ) const;

    void scan_one_batch(
        const char* batch_data,
        const char* ex_data,
        const PID* ids,
        const SplitBatchQuery<float>& q_obj,
        buffer::SearchBuffer<float>& knns,
        size_t num_points,
        bool
    ) const;

    void save_legacy(const char*) const;
    void save_v2(const char*) const;
    void load_legacy(const char*);
    void load_v2(const char*);
    void search_cluster_paged(
        PID, const SplitBatchQuery<float>&, buffer::SearchBuffer<float>&, bool
    ) const;
    static void write_fixed_page(
        std::ofstream&, uint32_t, uint32_t, uint32_t, const char*, size_t
    );
    static void read_fixed_page(
        std::ifstream&, uint32_t, IvfPageHeaderV2&, std::array<char, kPagePayloadV2>&
    );
    void open_mapped_file(const char*);
    void close_mapped_file();
    const IvfPageHeaderV2* mapped_page_header(uint32_t) const;
    PinnedPage pin_page(uint32_t, uint32_t) const;
    void unpin_page(uint32_t) const;
    void clear_page_cache() const;
    void evict_page_cache_locked(size_t) const;
    size_t read_page_span_mmap(uint32_t, uint32_t, uint32_t, std::vector<char>&) const;
    static size_t pages_for_bytes(size_t bytes) {
        return bytes == 0 ? 0 : div_round_up(bytes, kPagePayloadV2);
    }

   public:
    explicit IVF() {}
    explicit IVF(
        size_t,
        size_t,
        size_t,
        size_t,
        MetricType metric_type = rabitqlib::METRIC_L2,
        RotatorType type = RotatorType::FhtKacRotator
    );

    ~IVF();

    void construct(const float*, const float*, const PID*, bool);

    void save(const char*) const;
    void save_as_v2(const char*) const;

    void load(const char*);

    void search(const float*, size_t, size_t, PID*, bool) const;

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }

    [[nodiscard]] size_t num_clusters() const { return this->num_cluster_; }

    void set_page_cache_capacity_bytes(size_t bytes);
};

inline IVF::IVF(
    size_t n,
    size_t dim,
    size_t cluster_num,
    size_t bits,
    MetricType metric_type,
    RotatorType type
)
    : num_(n)
    , dim_(dim)
    , padded_dim_(dim)
    , num_cluster_(cluster_num)
    , ex_bits_(bits - 1)
    , type_(type)
    , metric_type_(metric_type) {
    if (bits < 1 || bits > 9) {
        std::cerr << "Invalid number of bits for quantization in IVF::IVF\n";
        std::cerr << "Expected: 1 to 9  Input:" << bits << '\n';
        std::cerr.flush();
        exit(1);
    };
    rotator_ = choose_rotator<float>(dim, type, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);
}

inline IVF::~IVF() {
    delete rotator_;
    free_memory();
}

inline IVF::PinnedPage::~PinnedPage() { release(); }

inline IVF::PinnedPage::PinnedPage(PinnedPage&& other) noexcept
    : owner_(other.owner_)
    , page_id_(other.page_id_)
    , next_page_id_(other.next_page_id_)
    , payload_(other.payload_)
    , payload_bytes_(other.payload_bytes_) {
    other.owner_ = nullptr;
    other.page_id_ = std::numeric_limits<uint32_t>::max();
    other.next_page_id_ = std::numeric_limits<uint32_t>::max();
    other.payload_ = nullptr;
    other.payload_bytes_ = 0;
}

inline IVF::PinnedPage& IVF::PinnedPage::operator=(PinnedPage&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        page_id_ = other.page_id_;
        next_page_id_ = other.next_page_id_;
        payload_ = other.payload_;
        payload_bytes_ = other.payload_bytes_;
        other.owner_ = nullptr;
        other.page_id_ = std::numeric_limits<uint32_t>::max();
        other.next_page_id_ = std::numeric_limits<uint32_t>::max();
        other.payload_ = nullptr;
        other.payload_bytes_ = 0;
    }
    return *this;
}

inline void IVF::PinnedPage::release() {
    if (owner_ != nullptr) {
        owner_->unpin_page(page_id_);
        owner_ = nullptr;
    }
    page_id_ = std::numeric_limits<uint32_t>::max();
    next_page_id_ = std::numeric_limits<uint32_t>::max();
    payload_ = nullptr;
    payload_bytes_ = 0;
}

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param clustter_ids Cluster ID for each data objects
 */
inline void IVF::construct(
    const float* data, const float* centroids, const PID* cluster_ids, bool faster = false
) {
    std::cout << "Start IVF construction...\n";

    // get id list for each cluster
    std::cout << "\tLoading clustering information...\n";
    std::vector<size_t> counts(num_cluster_, 0);
    std::vector<std::vector<PID>> id_lists(num_cluster_);
    for (size_t i = 0; i < num_; ++i) {
        PID cid = cluster_ids[i];
        if (cid > num_cluster_) {
            std::cerr << "Bad cluster id\n";
            exit(1);
        }
        id_lists[cid].push_back(static_cast<PID>(i));
        counts[cid] += 1;
    }

    allocate_memory(counts);

    // init the cluster list
    init_clusters(counts);

    // all rotated centroids
    std::vector<float> rotated_centroids(num_cluster_ * padded_dim_);

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    /* Quantize each cluster */
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_cluster_; ++i) {
        const float* cur_centroid = centroids + (i * dim_);
        float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
        Cluster& cp = cluster_lst_[i];
        quantize_cluster(cp, id_lists[i], data, cur_centroid, cur_rotated_c, config);
    }

    this->initer_->add_vectors(rotated_centroids.data());
}

inline void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    std::cout << "Allocating memory for IVF...\n";
    if (num_cluster_ < 20000UL) {
        this->initer_ = new FlatInitializer(padded_dim_, num_cluster_);
    } else {
        this->initer_ = new HNSWInitializer(padded_dim_, num_cluster_);
    }
    this->batch_data_ =
        memory::align_allocate<64, char, true>(batch_data_bytes(cluster_sizes));
    if (ex_bits_ > 0) {
        this->ex_data_ = memory::align_allocate<64, char, true>(ex_data_bytes());
    }
    this->ids_ = memory::align_allocate<64, PID, true>(ids_bytes());

    this->ip_func_ = select_excode_ipfunc(ex_bits_);
}

/**
 * @brief intialize the cluster list: finding idx for all data
 */
inline void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->cluster_lst_.reserve(num_cluster_);
    size_t added_vectors = 0;
    size_t added_batches = 0;
    for (size_t i = 0; i < num_cluster_; ++i) {
        // find data location for current cluster
        size_t num = cluster_sizes[i];
        size_t num_batches = div_round_up(num, fastscan::kBatchSize);

        char* current_batch_data =
            batch_data_ + (BatchDataMap<float>::data_bytes(padded_dim_) * added_batches);
        char* current_ex_data =
            ex_data_ +
            (added_vectors * ExDataMap<float>::data_bytes(padded_dim_, ex_bits_));
        PID* ids = ids_ + added_vectors;

        Cluster cur_cluster(num, current_batch_data, current_ex_data, ids);
        this->cluster_lst_.push_back(std::move(cur_cluster));

        added_vectors += num;
        added_batches += num_batches;
    }
}

inline void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& IDs,
    const float* data,
    const float* cur_centroid,
    float* rotated_centroid,
    const quant::RabitqConfig& config
) {
    size_t num_points = IDs.size();
    if (cp.num() != num_points) {
        std::cerr << "Size of cluster and IDs are inequivalent\n";
        std::cerr << "Cluster: " << cp.num() << " IDs: " << num_points << '\n';
        exit(1);
    }

    // copy ids
    std::copy(IDs.begin(), IDs.end(), cp.ids());

    // rotate centroid
    this->rotator_->rotate(cur_centroid, rotated_centroid);

    // rotate vectors for this cluster
    std::vector<float> rotated_data(padded_dim_ * num_points);
    for (size_t i = 0; i < num_points; ++i) {
        rotator_->rotate(data + (IDs[i] * dim_), rotated_data.data() + (i * padded_dim_));
    }

    char* batch_data = cp.batch_data();
    char* ex_data = cp.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t n = std::min(fastscan::kBatchSize, num_points - i);

        quant::quantize_split_batch(
            rotated_data.data() + (i * padded_dim_),
            rotated_centroid,
            n,
            padded_dim_,
            ex_bits_,
            batch_data,
            ex_data,
            metric_type_,
            config
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * n;
    }
}

inline void IVF::write_fixed_page(
    std::ofstream& out,
    uint32_t page_type,
    uint32_t page_id,
    uint32_t next_page_id,
    const char* payload,
    size_t payload_bytes
) {
    IvfPageHeaderV2 header;
    header.page_type = page_type;
    header.header_bytes = static_cast<uint16_t>(sizeof(IvfPageHeaderV2));
    header.flags = 0;
    header.page_id = page_id;
    header.next_page_id = next_page_id;
    header.payload_bytes = static_cast<uint32_t>(payload_bytes);
    header.crc32 = 0;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (payload_bytes > 0) {
        out.write(payload, static_cast<long>(payload_bytes));
    }
    if (payload_bytes < kPagePayloadV2) {
        std::array<char, kPagePayloadV2> zeros{};
        out.write(zeros.data(), static_cast<long>(kPagePayloadV2 - payload_bytes));
    }
}

inline void IVF::read_fixed_page(
    std::ifstream& in,
    uint32_t page_id,
    IvfPageHeaderV2& header,
    std::array<char, kPagePayloadV2>& payload
) {
    in.clear();
    const std::streamoff offset = static_cast<std::streamoff>(page_id) *
                                  static_cast<std::streamoff>(kPageSizeV2);
    in.seekg(offset, std::ios::beg);
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("Failed to read v2 page header");
    }
    if (header.header_bytes != sizeof(IvfPageHeaderV2)) {
        throw std::runtime_error("Invalid v2 page header size");
    }
    in.read(payload.data(), static_cast<long>(kPagePayloadV2));
    if (!in) {
        throw std::runtime_error("Failed to read v2 page payload");
    }
}

inline void IVF::open_mapped_file(const char* filename) {
    close_mapped_file();

    mapped_fd_ = ::open(filename, O_RDONLY);
    if (mapped_fd_ < 0) {
        throw std::runtime_error("Failed to open IVF v2 index file for mmap");
    }

    struct stat st {};
    if (::fstat(mapped_fd_, &st) != 0) {
        const int err = errno;
        ::close(mapped_fd_);
        mapped_fd_ = -1;
        throw std::runtime_error("fstat failed for IVF v2 index file: " + std::to_string(err));
    }
    if (st.st_size <= 0) {
        ::close(mapped_fd_);
        mapped_fd_ = -1;
        throw std::runtime_error("IVF v2 index file is empty");
    }

    mapped_size_ = static_cast<size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, mapped_fd_, 0);
    if (ptr == MAP_FAILED) {
        const int err = errno;
        ::close(mapped_fd_);
        mapped_fd_ = -1;
        mapped_size_ = 0;
        mapped_data_ = nullptr;
        throw std::runtime_error("mmap failed for IVF v2 index file: " + std::to_string(err));
    }
    mapped_data_ = static_cast<const char*>(ptr);
}

inline void IVF::close_mapped_file() {
    clear_page_cache();
    if (mapped_data_ != nullptr) {
        ::munmap(const_cast<char*>(mapped_data_), mapped_size_);
        mapped_data_ = nullptr;
        mapped_size_ = 0;
    }
    if (mapped_fd_ >= 0) {
        ::close(mapped_fd_);
        mapped_fd_ = -1;
    }
}

inline const IvfPageHeaderV2* IVF::mapped_page_header(uint32_t page_id) const {
    const size_t offset = static_cast<size_t>(page_id) * kPageSizeV2;
    if (offset + sizeof(IvfPageHeaderV2) > mapped_size_) {
        throw std::runtime_error("Page offset out of range in IVF v2 file");
    }
    auto header = reinterpret_cast<const IvfPageHeaderV2*>(mapped_data_ + offset);
    if (header->header_bytes != sizeof(IvfPageHeaderV2)) {
        throw std::runtime_error("Invalid v2 page header size");
    }
    if (header->payload_bytes > kPagePayloadV2) {
        throw std::runtime_error("Invalid v2 page payload size");
    }
    if (offset + sizeof(IvfPageHeaderV2) + header->payload_bytes > mapped_size_) {
        throw std::runtime_error("Invalid v2 page payload size");
    }
    return header;
}

inline IVF::PinnedPage IVF::pin_page(uint32_t page_id, uint32_t expected_page_type) const {
    {
        std::lock_guard<std::mutex> guard(page_cache_mu_);
        auto it = page_cache_index_.find(page_id);
        if (it != page_cache_index_.end()) {
            auto node_it = it->second;
            if (node_it->page_type != expected_page_type) {
                throw std::runtime_error("Unexpected page type in v2 file");
            }
            node_it->pin_count += 1;
            page_cache_lru_.splice(page_cache_lru_.begin(), page_cache_lru_, node_it);
            return PinnedPage(
                this,
                page_id,
                node_it->next_page_id,
                node_it->payload.data(),
                node_it->payload_bytes
            );
        }
    }

    const IvfPageHeaderV2* header = mapped_page_header(page_id);
    if (header->page_type != expected_page_type) {
        throw std::runtime_error("Unexpected page type in v2 file");
    }

    CachedPageV2 fresh;
    fresh.page_id = page_id;
    fresh.page_type = header->page_type;
    fresh.next_page_id = header->next_page_id;
    fresh.payload_bytes = header->payload_bytes;
    fresh.pin_count = 1;
    fresh.payload.resize(fresh.payload_bytes);
    if (fresh.payload_bytes > 0) {
        const char* payload = reinterpret_cast<const char*>(header) + sizeof(IvfPageHeaderV2);
        std::memcpy(fresh.payload.data(), payload, fresh.payload_bytes);
    }

    std::lock_guard<std::mutex> guard(page_cache_mu_);
    auto existing = page_cache_index_.find(page_id);
    if (existing != page_cache_index_.end()) {
        auto node_it = existing->second;
        if (node_it->page_type != expected_page_type) {
            throw std::runtime_error("Unexpected page type in v2 file");
        }
        node_it->pin_count += 1;
        page_cache_lru_.splice(page_cache_lru_.begin(), page_cache_lru_, node_it);
        return PinnedPage(
            this,
            page_id,
            node_it->next_page_id,
            node_it->payload.data(),
            node_it->payload_bytes
        );
    }

    evict_page_cache_locked(fresh.payload.size());
    page_cache_lru_.push_front(std::move(fresh));
    auto inserted = page_cache_lru_.begin();
    page_cache_index_[page_id] = inserted;
    page_cache_bytes_ += inserted->payload.size();
    return PinnedPage(
        this,
        page_id,
        inserted->next_page_id,
        inserted->payload.data(),
        inserted->payload_bytes
    );
}

inline void IVF::unpin_page(uint32_t page_id) const {
    std::lock_guard<std::mutex> guard(page_cache_mu_);
    auto it = page_cache_index_.find(page_id);
    if (it == page_cache_index_.end()) {
        return;
    }
    auto node_it = it->second;
    if (node_it->pin_count > 0) {
        node_it->pin_count -= 1;
    }
}

inline void IVF::clear_page_cache() const {
    std::lock_guard<std::mutex> guard(page_cache_mu_);
    page_cache_index_.clear();
    page_cache_lru_.clear();
    page_cache_bytes_ = 0;
}

inline void IVF::evict_page_cache_locked(size_t incoming_bytes) const {
    while (!page_cache_lru_.empty() &&
           page_cache_bytes_ + incoming_bytes > page_cache_capacity_bytes_) {
        bool evicted = false;
        for (auto it = page_cache_lru_.end(); it != page_cache_lru_.begin();) {
            --it;
            if (it->pin_count != 0) {
                continue;
            }
            page_cache_bytes_ -= it->payload.size();
            page_cache_index_.erase(it->page_id);
            page_cache_lru_.erase(it);
            evicted = true;
            break;
        }
        if (!evicted) {
            break;
        }
    }
}

inline size_t IVF::read_page_span_mmap(
    uint32_t first_page_id,
    uint32_t page_count,
    uint32_t expected_page_type,
    std::vector<char>& out
) const {
    out.clear();
    if (page_count == 0) {
        return 0;
    }
    std::vector<PinnedPage> pins;
    pins.reserve(page_count);
    out.reserve(page_count * kPagePayloadV2);
    uint32_t cur_page = first_page_id;
    for (uint32_t i = 0; i < page_count; ++i) {
        pins.push_back(pin_page(cur_page, expected_page_type));
        const PinnedPage& page = pins.back();
        if (page.payload_bytes() > 0) {
            out.insert(out.end(), page.payload(), page.payload() + page.payload_bytes());
        }
        cur_page = page.next_page_id();
        if (i + 1 != page_count &&
            cur_page == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Broken page chain in v2 file");
        }
    }
    return out.size();
}

inline void IVF::save(const char* filename) const { save_legacy(filename); }

inline void IVF::save_as_v2(const char* filename) const { save_v2(filename); }

inline void IVF::save_legacy(const char* filename) const {
    if (cluster_lst_.empty()) {
        std::cerr << "IVF not constructed\n";
        return;
    }

    std::ofstream output(filename, std::ios::binary);

    output.write(reinterpret_cast<const char*>(&num_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&type_), sizeof(type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cluster_);
    for (const auto& cur_cluster : cluster_lst_) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * num_cluster_)
    );

    this->rotator_->save(output);
    this->initer_->save(output, filename);
    output.write(
        reinterpret_cast<const char*>(batch_data_),
        static_cast<long>(batch_data_bytes(cluster_sizes))
    );
    output.write(
        reinterpret_cast<const char*>(ex_data_), static_cast<long>(ex_data_bytes())
    );
    output.write(reinterpret_cast<const char*>(ids_), static_cast<long>(ids_bytes()));
}

inline void IVF::save_v2(const char* filename) const {
    if (cluster_lst_.empty()) {
        std::cerr << "IVF not constructed\n";
        return;
    }

    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cluster_);
    for (const auto& cur_cluster : cluster_lst_) {
        cluster_sizes.push_back(cur_cluster.num());
    }

    std::vector<char> rotator_blob(rotator_->dump_bytes());
    rotator_->save(rotator_blob.data());

    std::vector<float> centroids(num_cluster_ * padded_dim_);
    for (size_t i = 0; i < num_cluster_; ++i) {
        std::memcpy(
            centroids.data() + (i * padded_dim_),
            initer_->centroid(static_cast<PID>(i)),
            sizeof(float) * padded_dim_
        );
    }
    const char* centroids_ptr = reinterpret_cast<const char*>(centroids.data());
    const size_t centroids_bytes = centroids.size() * sizeof(float);

    std::vector<std::vector<char>> cluster_blobs(num_cluster_);
    std::vector<IvfClusterDirEntryV2> dir(num_cluster_);

    const size_t ex_one_bytes = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    size_t running_batch_ordinal = 0;
    uint32_t cur_cluster_page = 1 +
                                static_cast<uint32_t>(pages_for_bytes(rotator_blob.size())) +
                                static_cast<uint32_t>(pages_for_bytes(centroids_bytes)) +
                                static_cast<uint32_t>(pages_for_bytes(
                                    sizeof(IvfClusterDirEntryV2) * num_cluster_
                                ));

    for (size_t cid = 0; cid < num_cluster_; ++cid) {
        const Cluster& c = cluster_lst_[cid];
        const size_t num = c.num();
        const size_t num_batches = div_round_up(num, fastscan::kBatchSize);
        const size_t batch_bytes = num_batches * BatchDataMap<float>::data_bytes(padded_dim_);
        const size_t ex_bytes = num * ex_one_bytes;
        const size_t ids_bytes = num * sizeof(PID);
        const size_t total_bytes = batch_bytes + ex_bytes + ids_bytes;

        cluster_blobs[cid].resize(total_bytes);
        char* blob = cluster_blobs[cid].data();
        if (batch_bytes > 0) {
            std::memcpy(blob, c.batch_data(), batch_bytes);
        }
        if (ex_bytes > 0) {
            std::memcpy(blob + batch_bytes, c.ex_data(), ex_bytes);
        }
        if (ids_bytes > 0) {
            std::memcpy(blob + batch_bytes + ex_bytes, c.ids(), ids_bytes);
        }

        IvfClusterDirEntryV2 entry;
        entry.cluster_id = static_cast<uint32_t>(cid);
        entry.num_points = static_cast<uint32_t>(num);
        entry.num_batches = static_cast<uint32_t>(num_batches);
        entry.flags = (ex_bits_ > 0) ? 0x2U : 0U;
        entry.first_page_id = cur_cluster_page;
        const uint32_t cluster_pages = static_cast<uint32_t>(pages_for_bytes(total_bytes));
        entry.last_page_id =
            cluster_pages == 0 ? cur_cluster_page : cur_cluster_page + cluster_pages - 1;
        entry.first_batch_ordinal = static_cast<uint32_t>(running_batch_ordinal);
        entry.cluster_bytes = static_cast<uint64_t>(total_bytes);
        entry.batch_bytes = static_cast<uint64_t>(batch_bytes);
        entry.ex_bytes = static_cast<uint64_t>(ex_bytes);
        entry.ids_bytes = static_cast<uint64_t>(ids_bytes);

        dir[cid] = entry;
        cur_cluster_page += cluster_pages;
        running_batch_ordinal += num_batches;
    }

    const size_t rotator_pages = pages_for_bytes(rotator_blob.size());
    const size_t centroid_pages = pages_for_bytes(centroids_bytes);
    const size_t dir_bytes = sizeof(IvfClusterDirEntryV2) * num_cluster_;
    const size_t dir_pages = pages_for_bytes(dir_bytes);
    size_t cluster_pages = 0;
    for (const auto& blob : cluster_blobs) {
        cluster_pages += pages_for_bytes(blob.size());
    }

    IvfMetaV2 meta;
    std::memcpy(meta.magic, kIvfMagicV2, sizeof(kIvfMagicV2));
    meta.format_version = kFormatV2;
    meta.page_size = static_cast<uint32_t>(kPageSizeV2);
    meta.num_points = static_cast<uint64_t>(num_);
    meta.dim = static_cast<uint32_t>(dim_);
    meta.padded_dim = static_cast<uint32_t>(padded_dim_);
    meta.num_clusters = static_cast<uint32_t>(num_cluster_);
    meta.ex_bits = static_cast<uint32_t>(ex_bits_);
    meta.metric_type = static_cast<uint32_t>(metric_type_);
    meta.rotator_type = static_cast<uint32_t>(type_);
    meta.init_type = dynamic_cast<HNSWInitializer*>(initer_) != nullptr ? 1U : 0U;
    meta.rotator_first_page = 1;
    meta.rotator_page_count = static_cast<uint32_t>(rotator_pages);
    meta.centroid_first_page = meta.rotator_first_page + meta.rotator_page_count;
    meta.centroid_page_count = static_cast<uint32_t>(centroid_pages);
    meta.dir_first_page = meta.centroid_first_page + meta.centroid_page_count;
    meta.dir_page_count = static_cast<uint32_t>(dir_pages);
    meta.cluster_first_page = meta.dir_first_page + meta.dir_page_count;
    meta.cluster_page_count = static_cast<uint32_t>(cluster_pages);

    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file for IVF v2");
    }

    write_fixed_page(
        out,
        kIvfPageTypeMeta,
        0,
        std::numeric_limits<uint32_t>::max(),
        reinterpret_cast<const char*>(&meta),
        sizeof(meta)
    );

    auto write_blob_pages = [&](uint32_t page_type, uint32_t first_page_id, const char* data, size_t bytes) {
        size_t written = 0;
        uint32_t page_id = first_page_id;
        while (written < bytes) {
            const size_t chunk = std::min(kPagePayloadV2, bytes - written);
            const uint32_t next_page =
                (written + chunk < bytes) ? (page_id + 1) : std::numeric_limits<uint32_t>::max();
            write_fixed_page(out, page_type, page_id, next_page, data + written, chunk);
            written += chunk;
            ++page_id;
        }
    };

    if (!rotator_blob.empty()) {
        write_blob_pages(
            kIvfPageTypeRotator,
            meta.rotator_first_page,
            rotator_blob.data(),
            rotator_blob.size()
        );
    }
    if (centroids_bytes > 0) {
        write_blob_pages(
            kIvfPageTypeCentroids, meta.centroid_first_page, centroids_ptr, centroids_bytes
        );
    }
    if (dir_bytes > 0) {
        write_blob_pages(
            kIvfPageTypeDirectory,
            meta.dir_first_page,
            reinterpret_cast<const char*>(dir.data()),
            dir_bytes
        );
    }

    for (size_t cid = 0; cid < num_cluster_; ++cid) {
        const auto& blob = cluster_blobs[cid];
        if (blob.empty()) {
            continue;
        }
        write_blob_pages(
            kIvfPageTypeClusterData, dir[cid].first_page_id, blob.data(), blob.size()
        );
    }
}

inline void IVF::load(const char* filename) {
    std::ifstream probe(filename, std::ios::binary);
    if (!probe.is_open()) {
        throw std::runtime_error("Failed to open IVF index file");
    }

    char magic[8] = {};
    probe.read(magic, sizeof(magic));
    if (!probe) {
        throw std::runtime_error("Failed to read IVF index header");
    }
    probe.close();

    if (std::memcmp(magic, kIvfMagicV2, sizeof(kIvfMagicV2)) == 0) {
        load_v2(filename);
    } else {
        load_legacy(filename);
    }
}

inline void IVF::load_legacy(const char* filename) {
    std::cout << "Loading IVF (legacy format)...\n";
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open IVF index file");
    }

    free_memory();
    delete rotator_;
    rotator_ = nullptr;

    input.read(reinterpret_cast<char*>(&this->num_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->ex_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&type_), sizeof(type_));
    input.read(reinterpret_cast<char*>(&metric_type_), sizeof(metric_type_));

    rotator_ = choose_rotator<float>(dim_, type_, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();

    std::vector<size_t> cluster_sizes(num_cluster_, 0);
    input.read(
        reinterpret_cast<char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * num_cluster_)
    );

    size_t tmp =
        std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0));
    if (tmp != num_) {
        throw std::runtime_error("The sum of cluster num != total number of points");
    }

    this->rotator_->load(input);
    allocate_memory(cluster_sizes);
    this->initer_->load(input, filename);
    input.read(batch_data_, static_cast<long>(batch_data_bytes(cluster_sizes)));
    input.read(ex_data_, static_cast<long>(ex_data_bytes()));
    input.read(reinterpret_cast<char*>(ids_), static_cast<long>(ids_bytes()));
    init_clusters(cluster_sizes);

    this->loaded_index_path_ = filename;
    this->storage_mode_ = StorageMode::LegacyInMemory;
    std::cout << "Index loaded\n";
}

inline void IVF::load_v2(const char* filename) {
    std::cout << "Loading IVF (v2 paged format)...\n";
    free_memory();
    delete rotator_;
    rotator_ = nullptr;

    open_mapped_file(filename);

    const IvfPageHeaderV2* meta_header = mapped_page_header(0);
    const char* meta_payload = reinterpret_cast<const char*>(meta_header) + sizeof(IvfPageHeaderV2);
    if (meta_header->page_type != kIvfPageTypeMeta) {
        throw std::runtime_error("Invalid IVF v2 meta page type");
    }
    if (meta_header->payload_bytes < sizeof(IvfMetaV2)) {
        throw std::runtime_error("Invalid IVF v2 meta payload");
    }

    IvfMetaV2 meta{};
    std::memcpy(&meta, meta_payload, sizeof(meta));
    if (std::memcmp(meta.magic, kIvfMagicV2, sizeof(kIvfMagicV2)) != 0) {
        throw std::runtime_error("Invalid IVF v2 magic");
    }
    if (meta.format_version != kFormatV2) {
        throw std::runtime_error("Unsupported IVF v2 format version");
    }
    if (meta.page_size != kPageSizeV2) {
        throw std::runtime_error("Unsupported IVF v2 page size");
    }

    num_ = static_cast<size_t>(meta.num_points);
    dim_ = static_cast<size_t>(meta.dim);
    padded_dim_ = static_cast<size_t>(meta.padded_dim);
    num_cluster_ = static_cast<size_t>(meta.num_clusters);
    ex_bits_ = static_cast<size_t>(meta.ex_bits);
    metric_type_ = static_cast<MetricType>(meta.metric_type);
    type_ = static_cast<RotatorType>(meta.rotator_type);
    ip_func_ = select_excode_ipfunc(ex_bits_);

    rotator_ = choose_rotator<float>(dim_, type_, round_up_to_multiple(dim_, 64));

    std::vector<char> blob;
    read_page_span_mmap(
        meta.rotator_first_page,
        meta.rotator_page_count,
        kIvfPageTypeRotator,
        blob
    );
    if (blob.size() != rotator_->dump_bytes()) {
        throw std::runtime_error("Invalid IVF v2 rotator payload");
    }
    rotator_->load(blob.data());
    padded_dim_ = rotator_->size();

    read_page_span_mmap(
        meta.centroid_first_page,
        meta.centroid_page_count,
        kIvfPageTypeCentroids,
        blob
    );
    const size_t expected_centroids_bytes = num_cluster_ * padded_dim_ * sizeof(float);
    if (blob.size() != expected_centroids_bytes) {
        throw std::runtime_error("Invalid IVF v2 centroid payload");
    }

    if (meta.init_type == 1) {
        initer_ = new HNSWInitializer(padded_dim_, num_cluster_);
    } else {
        initer_ = new FlatInitializer(padded_dim_, num_cluster_);
    }
    initer_->add_vectors(reinterpret_cast<const float*>(blob.data()));

    read_page_span_mmap(
        meta.dir_first_page,
        meta.dir_page_count,
        kIvfPageTypeDirectory,
        blob
    );
    const size_t expected_dir_bytes = num_cluster_ * sizeof(IvfClusterDirEntryV2);
    if (blob.size() != expected_dir_bytes) {
        throw std::runtime_error("Invalid IVF v2 directory payload");
    }
    cluster_dir_v2_.resize(num_cluster_);
    std::memcpy(cluster_dir_v2_.data(), blob.data(), expected_dir_bytes);

    loaded_index_path_ = filename;
    storage_mode_ = StorageMode::PagedV2;
    std::cout << "Index loaded\n";
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    bool use_hacc = true
) const {
    nprobe = std::min(nprobe, num_cluster_);  // corner case
    std::vector<float> rotated_query(padded_dim_);
    this->rotator_->rotate(query, rotated_query.data());

    // use initer to get closest nprobe centroids
    std::vector<AnnCandidate<float>> centroid_dist(nprobe);
    this->initer_->centroids_distances(rotated_query.data(), nprobe, centroid_dist);

    buffer::SearchBuffer knns(k);

    SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        float dist = centroid_dist[i].distance;
        if (metric_type_ == METRIC_L2) {
            q_obj.set_g_add(dist);
        } else if (metric_type_ == METRIC_IP) {
            auto g_add_ip = dot_product<float>(
                rotated_query.data(), initer_->centroid(cid), padded_dim_
            );
            q_obj.set_g_add(dist, g_add_ip);
        } else {
            // unsupported
            std::cerr << "Invalid quantize metric type, only support L2 and IP metric\n "
                      << std::flush;
            return;
        }
        if (storage_mode_ == StorageMode::PagedV2) {
            search_cluster_paged(cid, q_obj, knns, use_hacc);
        } else {
            const Cluster& cur_cluster = cluster_lst_[cid];
            search_cluster(cur_cluster, q_obj, knns, use_hacc);
        }
    }

    knns.copy_results(results);
}

inline void IVF::search_cluster_paged(
    PID cid,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    bool use_hacc
) const {
    if (cid >= cluster_dir_v2_.size()) {
        return;
    }
    const IvfClusterDirEntryV2& entry = cluster_dir_v2_[cid];
    if (entry.cluster_bytes == 0 || entry.num_points == 0) {
        return;
    }

    std::vector<char> blob;
    read_page_span_mmap(
        entry.first_page_id,
        static_cast<uint32_t>(entry.last_page_id - entry.first_page_id + 1),
        kIvfPageTypeClusterData,
        blob
    );
    if (blob.size() < entry.cluster_bytes) {
        throw std::runtime_error("Short cluster payload in IVF v2");
    }

    const size_t batch_bytes = static_cast<size_t>(entry.batch_bytes);
    const size_t ex_bytes = static_cast<size_t>(entry.ex_bytes);
    const size_t num_points = static_cast<size_t>(entry.num_points);
    const size_t ids_bytes = static_cast<size_t>(entry.ids_bytes);
    if (batch_bytes + ex_bytes + ids_bytes > blob.size()) {
        throw std::runtime_error("Corrupted cluster offsets in IVF v2");
    }

    std::vector<PID> ids(num_points);
    if (ids_bytes > 0) {
        std::memcpy(ids.data(), blob.data() + batch_bytes + ex_bytes, ids_bytes);
    }

    const char* batch_data = blob.data();
    const char* ex_data = blob.data() + batch_bytes;
    const PID* id_ptr = ids.data();

    size_t iter = num_points / fastscan::kBatchSize;
    size_t remain = num_points - (iter * fastscan::kBatchSize);
    for (size_t i = 0; i < iter; ++i) {
        scan_one_batch(
            batch_data, ex_data, id_ptr, q_obj, knns, fastscan::kBatchSize, use_hacc
        );
        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        ex_data +=
            ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * fastscan::kBatchSize;
        id_ptr += fastscan::kBatchSize;
    }
    if (remain > 0) {
        scan_one_batch(batch_data, ex_data, id_ptr, q_obj, knns, remain, use_hacc);
    }
}

inline void IVF::search_cluster(
    const Cluster& cur_cluster,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    bool use_hacc
) const {
    size_t iter = cur_cluster.num() / fastscan::kBatchSize;
    size_t remain = cur_cluster.num() - (iter * fastscan::kBatchSize);

    const char* batch_data = cur_cluster.batch_data();
    const char* ex_data = cur_cluster.ex_data();
    const PID* ids = cur_cluster.ids();

    /* Compute distances block by block */
    for (size_t i = 0; i < iter; ++i) {
        scan_one_batch(
            batch_data, ex_data, ids, q_obj, knns, fastscan::kBatchSize, use_hacc
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        ex_data +=
            ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * fastscan::kBatchSize;
        ids += fastscan::kBatchSize;
    }

    if (remain > 0) {
        // scan the last block
        scan_one_batch(batch_data, ex_data, ids, q_obj, knns, remain, use_hacc);
    }
}

inline void IVF::set_page_cache_capacity_bytes(size_t bytes) {
    std::lock_guard<std::mutex> guard(page_cache_mu_);
    page_cache_capacity_bytes_ = bytes;
    evict_page_cache_locked(0);
}

inline void IVF::scan_one_batch(
    const char* batch_data,
    const char* ex_data,
    const PID* ids,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    size_t num_points,
    bool use_hacc
) const {
    std::array<float, fastscan::kBatchSize> est_distance;  // estimated distance
    std::array<float, fastscan::kBatchSize> low_distance;  // lower distance
    std::array<float, fastscan::kBatchSize> ip_x0_qr;      // inner product of the 1st bit

    split_batch_estdist(
        batch_data,
        q_obj,
        padded_dim_,
        est_distance.data(),
        low_distance.data(),
        ip_x0_qr.data(),
        use_hacc
    );

    float distk = knns.top_dist();

    // if only use 1-bit code, directly return
    if (ex_bits_ == 0) {
        for (size_t i = 0; i < num_points; ++i) {
            PID id = ids[i];
            float ex_dist = est_distance[i];
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        return;
    }

    // incremental distance computation - V2
    for (size_t i = 0; i < num_points; ++i) {
        float lower_dist = low_distance[i];
        if (lower_dist < distk) {
            PID id = ids[i];
            ConstExDataMap<float> cur_ex(ex_data, padded_dim_, ex_bits_);
            float ex_dist = split_distance_boosting(
                ex_data, ip_func_, q_obj, padded_dim_, ex_bits_, ip_x0_qr[i]
            );
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    }
}
}  // namespace rabitqlib::ivf
