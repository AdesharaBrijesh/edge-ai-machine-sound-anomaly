/* Batch evaluation of the exported impulse on whole WAV clips (16 kHz, mono, 16-bit).
 *
 * Drop-in replacement for source/custom.cpp in example-standalone-inferencing-linux
 * (see eval.sh). Slides the impulse window over each clip with the impulse's stride
 * (1000 ms / 500 ms) and prints one CSV line per window:
 *
 *   file,window,<label 1>,<label 2>,...,anomaly,dsp_ms,classification_ms
 *
 *   ./build/custom clip1.wav clip2.wav ...      (or a single list file: @files.txt)
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#ifndef EVAL_STRIDE_MS
#define EVAL_STRIDE_MS 500
#endif

static bool read_wav(const std::string &path, std::vector<float> &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) || memcmp(buf.data() + 8, "WAVE", 4)) return false;
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    size_t p = 12;
    while (p + 8 <= buf.size()) {
        uint32_t size;
        memcpy(&size, buf.data() + p + 4, 4);
        if (!memcmp(buf.data() + p, "fmt ", 4)) {
            memcpy(&format, buf.data() + p + 8, 2);
            memcpy(&channels, buf.data() + p + 10, 2);
            memcpy(&rate, buf.data() + p + 12, 4);
            memcpy(&bits, buf.data() + p + 22, 2);
        } else if (!memcmp(buf.data() + p, "data", 4)) {
            if (format != 1 || channels != 1 || bits != 16 || rate != EI_CLASSIFIER_FREQUENCY) {
                fprintf(stderr, "%s: need 16-bit mono PCM at %d Hz\n", path.c_str(), EI_CLASSIFIER_FREQUENCY);
                return false;
            }
            size_t n = std::min<size_t>(size, buf.size() - p - 8) / 2;
            out.resize(n);
            for (size_t i = 0; i < n; i++) {
                int16_t s;
                memcpy(&s, buf.data() + p + 8 + 2 * i, 2);
                out[i] = (float)s;   // Edge Impulse raw audio features are int16-range values
            }
            return true;
        }
        p += 8 + size + (size & 1);
    }
    return false;
}

int main(int argc, char **argv) {
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '@') {
            std::ifstream list(argv[i] + 1);
            for (std::string line; std::getline(list, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) files.push_back(line);
            }
        } else {
            files.push_back(argv[i]);
        }
    }
    if (files.empty()) {
        printf("usage: %s clip.wav [...] | @list.txt\n", argv[0]);
        return 1;
    }

    run_classifier_init();
    const size_t win = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    const size_t hop = (size_t)EI_CLASSIFIER_FREQUENCY * EVAL_STRIDE_MS / 1000;

    printf("file,window");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) printf(",%s", ei_classifier_inferencing_categories[ix]);
    printf(",anomaly,dsp_ms,classification_ms\n");

    for (const auto &path : files) {
        std::vector<float> samples;
        if (!read_wav(path, samples)) {
            fprintf(stderr, "skipping %s\n", path.c_str());
            continue;
        }
        for (size_t start = 0, w = 0; start + win <= samples.size(); start += hop, w++) {
            signal_t signal;
            numpy::signal_from_buffer(&samples[start], win, &signal);
            ei_impulse_result_t result = { 0 };
            EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
            if (res != EI_IMPULSE_OK) {
                fprintf(stderr, "%s window %zu: run_classifier failed (%d)\n", path.c_str(), w, (int)res);
                return 1;
            }
            printf("%s,%zu", path.c_str(), w);
            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) printf(",%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY
            printf(",%.4f", result.anomaly);
#else
            printf(",");
#endif
            printf(",%d,%d\n", result.timing.dsp, result.timing.classification);
        }
    }
    return 0;
}
