// Turns a folder of 16x16 pictures into a dataset file, the same import the Datasets tab does.
// Usage: import_dataset <folder of pictures> <dataset name> <folder of datasets>
#include "data/dataset_store.h"
#include "data/file_names.h"
#include "data/sprite_importer.h"

#include <chrono>
#include <cstdio>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: import_dataset <folder of pictures> <dataset name> <folder of datasets>\n");
        return 2;
    }
    const std::string name = data::trimmed(argv[2]);
    if (!data::valid_file_name(name)) {
        std::fprintf(stderr, "not a valid dataset name: %s\n", name.c_str());
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    data::SpriteImporter importer(data::from_utf8(argv[1]), {255, 0, 255});
    while (!importer.finished()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::printf("\r%zu / %zu", importer.done(), importer.total());
        std::fflush(stdout);
    }
    data::ImportResult result = importer.take_result();
    if (!result.error.empty()) {
        std::fprintf(stderr, "\nimport failed: %s\n", result.error.c_str());
        return 1;
    }
    data::DatasetStore(data::from_utf8(argv[3])).save(name, result.dataset);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("\n%zu sprites, %zu skipped, %.1f s\n", result.dataset.count(), result.skipped, seconds);
    return 0;
}
