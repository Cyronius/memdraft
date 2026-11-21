// Experimental native counterpart of ple_experiment.py. GGUF rows are mmap'd
// from the original files, sharing the OS page cache with the target model.
// Linux/POSIX reference implementation; optimize only after measuring it.
#pragma once
#include "memdraft.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace memdraft {
struct ple_reranker {
    uint32_t K = 0, d = 0, with_ple = 0, nh = 0, nm = 0, per = 0, eos = 0;
    uint32_t base_width = 0, width = 0, V = 0, nheads = 0;
    std::vector<uint64_t> mult, offsets, vocabs;
    std::vector<uint32_t> heads;
    std::vector<float> freq, pin, pout, mean, scale;
    std::array<std::vector<float>, 4> w, bias;

    struct mapping {
        void * data = MAP_FAILED;
        uint64_t size = 0, offset = 0;
        uint32_t width = 0, rows = 0, type = 0;
        mapping() = default;
        mapping(const mapping &) = delete;
        mapping & operator=(const mapping &) = delete;
        ~mapping() { if (data != MAP_FAILED) munmap(data, size); }
        std::vector<float> row(uint32_t id) const {
            if (id >= rows) throw std::runtime_error("PLE token/row out of bounds");
            auto t = (ggml_type) type;
            const auto * src = (const char *) data + offset + (uint64_t) id * ggml_row_size(t, width);
            std::vector<float> result(width);
            if (t == GGML_TYPE_F32) std::memcpy(result.data(), src, width * sizeof(float));
            else ggml_get_type_traits(t)->to_float(src, result.data(), width);
            return result;
        }
    } inp, out, ple;

    template<class T> static void read(std::istream & f, T & x) {
        if (!f.read((char *) &x, sizeof(x))) throw std::runtime_error("truncated PLE scorer");
    }
    template<class T> static void read_vec(std::istream & f, std::vector<T> & v, size_t n) {
        v.resize(n);
        if (!f.read((char *) v.data(), n * sizeof(T))) throw std::runtime_error("truncated PLE scorer array");
    }
    static void map_tensor(std::istream & f, mapping & m) {
        uint32_t n;
        read(f, n);
        if (n == 0 || n > 16384) throw std::runtime_error("invalid GGUF path length");
        std::string path(n, '\0');
        if (!f.read(&path[0], n)) throw std::runtime_error("truncated GGUF path");
        read(f, m.size); read(f, m.offset); read(f, m.width); read(f, m.rows); read(f, m.type);
        if (!m.width || m.width > 65536 || !m.rows || m.type >= GGML_TYPE_COUNT)
            throw std::runtime_error("invalid GGUF tensor descriptor");
        const auto * traits = ggml_get_type_traits((ggml_type) m.type);
        if (!traits->blck_size || m.width % traits->blck_size ||
            (m.type != GGML_TYPE_F32 && !traits->to_float)) throw std::runtime_error("unsupported tensor type");
        const uint64_t bytes = (uint64_t) m.rows * ggml_row_size((ggml_type) m.type, m.width);
        if (m.offset > m.size || bytes > m.size - m.offset) throw std::runtime_error("tensor exceeds GGUF");
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open source GGUF: " + path);
        struct stat st;
        bool ok = fstat(fd, &st) == 0 && (uint64_t) st.st_size == m.size;
        if (ok) m.data = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (!ok || m.data == MAP_FAILED) throw std::runtime_error("GGUF changed or mmap failed: " + path);
    }
    bool load(const std::string & path, std::string & err) {
        try {
            std::ifstream f(path, std::ios::binary);
            char magic[8];
            if (!f.read(magic, 8) || std::memcmp(magic, "MDPLE001", 8)) throw std::runtime_error("invalid PLE scorer magic");
            for (auto * p : {&K, &d, &with_ple, &nh, &nm, &per, &eos, &base_width, &width, &V, &nheads}) read(f, *p);
            if (K < 3 || K > K_MAX || d != 64 || with_ple > 1 || !nh || nh > 64 || nm < 2 || nm > K ||
                !per || nheads != (nm - 1) * per || nheads > 64 || V > 1048576 || V < 1 || width > 65536 ||
                base_width != 8 * 2 * K + 9 + (K + 1) * d) throw std::runtime_error("invalid PLE scorer dimensions");
            read_vec(f, mult, nm); read_vec(f, offsets, nheads); read_vec(f, vocabs, nheads); read_vec(f, heads, nh);
            map_tensor(f, inp); map_tensor(f, out); map_tensor(f, ple);
            if (inp.rows != V || out.rows != V || width != base_width + with_ple * 2 * nh * ple.width)
                throw std::runtime_error("scorer/tensor dimensions disagree");
            for (uint32_t h = 0; h < nheads; ++h)
                if (!vocabs[h] || offsets[h] > ple.rows || vocabs[h] > ple.rows - offsets[h])
                    throw std::runtime_error("invalid PLE head range");
            for (uint32_t h : heads) if (h >= nheads || vocabs[h] <= 1) throw std::runtime_error("selected PLE head absent");
            read_vec(f, freq, V); read_vec(f, pin, (size_t) inp.width * d); read_vec(f, pout, (size_t) out.width * d);
            read_vec(f, mean, width); read_vec(f, scale, width);
            for (float s : scale) if (!(s > 0) || !std::isfinite(s)) throw std::runtime_error("invalid feature scale");
            uint32_t ni = width;
            for (uint32_t l = 0; l < 4; ++l) {
                uint32_t no = l == 0 ? 512 : l == 1 ? 256 : l == 2 ? 128 : 1;
                read_vec(f, w[l], (size_t) ni * no); read_vec(f, bias[l], no);
                ni = no;
            }
            return true;
        } catch (const std::exception & e) { err = e.what(); return false; }
    }
    void append_token(std::vector<float> & x, const mapping & table, const std::vector<float> & proj, int32_t t) const {
        auto raw = table.row(t < 0 ? 0 : (uint32_t) t);
        const size_t at = x.size();
        x.resize(at + d, 0.0f);
        for (uint32_t j = 0; j < table.width; ++j)
            for (uint32_t c = 0; c < d; ++c) x[at + c] += raw[j] * proj[(size_t) j * d + c];
    }
    void append_ple(std::vector<float> & x, const int32_t * ctx, uint32_t avail, int32_t candidate) const {
        std::vector<uint64_t> products(nm), mixed(nm);
        bool cut = false;
        for (uint32_t j = 0; j < nm; ++j) {
            int32_t t = candidate >= 0 ? (j == 0 ? candidate : (j - 1 < avail ? ctx[j - 1] : -1))
                                       : (j < avail ? ctx[j] : -1);
            if (j) cut = cut || t < 0 || (uint32_t) t == eos;
            if (cut || t < 0) t = eos;
            products[j] = (uint64_t) t * mult[j];
            mixed[j] = products[j] ^ (j ? mixed[j - 1] : 0);
        }
        for (uint32_t h : heads) {
            const uint32_t n = h / per + 1;
            auto row = ple.row((uint32_t) (mixed[n] % vocabs[h] + offsets[h]));
            x.insert(x.end(), row.begin(), row.end());
        }
    }
    std::vector<float> features(const proposals & p, const int32_t * ctx, uint32_t avail, int32_t token) const {
        const uint32_t S = 2 * K;
        std::vector<float> x(8 * S + 9, 0.0f);
        float own = 0, rival = 0, nag = 0, top = 0;
        int base = -1;
        for (uint32_t s = 0; s < S; ++s) if (p.CF[s] >= std::max(top, 1e-9f)) { top = p.CF[s]; base = s; }
        for (uint32_t s = 0; s < S; ++s) {
            const bool ag = p.H[s] && p.T[s] == token;
            if (ag) {
                x[s] = 1; x[S + s] = p.CF[s]; x[2*S + s] = p.A[s]; x[3*S + s] = p.B[s];
                own = std::max(own, p.CF[s]); nag += 1;
            } else rival = std::max(rival, p.CF[s]);
            x[4*S + s] = p.H[s]; x[5*S + s] = p.CF[s]; x[6*S + s] = p.A[s]; x[7*S + s] = p.B[s];
        }
        size_t at = 8 * S;
        x[at] = own; x[at+1] = rival; x[at+2] = nag/S; x[at+3] = own-rival;
        x[at+4] = base >= 0 && p.T[base] == token;
        x[at+5] = freq.at(token); x[at+6] = freq.at(ctx[0]); x[at+7] = token == ctx[0]; x[at+8] = token == ctx[1];
        append_token(x, out, pout, token);
        for (uint32_t c = 0; c < K; ++c) append_token(x, inp, pin, c < avail ? ctx[c] : 0);
        if (with_ple) { append_ple(x, ctx, avail, -1); append_ple(x, ctx, avail, token); }
        return x;
    }
    float score(std::vector<float> x) const {
        for (size_t i = 0; i < x.size(); ++i) x[i] = (x[i] - mean[i]) / scale[i];
        for (uint32_t l = 0; l < 4; ++l) {
            auto next = bias[l];
            for (size_t o = 0; o < next.size(); ++o)
                for (size_t i = 0; i < x.size(); ++i) next[o] += w[l][o*x.size() + i] * x[i];
            if (l < 3) for (float & v : next) v = std::max(v, 0.0f);
            x.swap(next);
        }
        return x[0];
    }
    int32_t pick(const proposals & p, const int32_t * ctx, uint32_t avail, float & prob) const {
        int32_t best = -1;
        float z = -INFINITY;
        for (uint32_t s = 0; s < p.S; ++s) {
            if (!p.H[s]) continue;
            bool duplicate = false;
            for (uint32_t r = 0; r < s; ++r) duplicate |= p.H[r] && p.T[r] == p.T[s];
            if (duplicate) continue;
            float v = score(features(p, ctx, avail, p.T[s]));
            if (v >= z) { z = v; best = p.T[s]; }
        }
        prob = best < 0 ? 0 : 1 / (1 + std::exp(-z));
        return best;
    }
    static int32_t callback(void * self, const proposals & p, const int32_t * ctx, uint32_t avail, float & prob) {
        return ((ple_reranker *) self)->pick(p, ctx, avail, prob);
    }
};
} // namespace memdraft
