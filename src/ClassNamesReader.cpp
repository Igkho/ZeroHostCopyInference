#include "ClassNamesReader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <yaml-cpp/yaml.h> // Implementation detail, not exposed in header

namespace cropandweed {

std::map<int, std::string> ClassNamesReader::Read(const std::string& filePath) {
    std::map<int, std::string> names;
    
    if (filePath.empty() || !std::filesystem::exists(filePath)) {
        std::cerr << "[Warning] Class file not found: " << filePath << std::endl;
        return names;
    }

    std::string ext = std::filesystem::path(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    try {
        if (ext == ".yaml" || ext == ".yml") {
            return ParseYaml(filePath);
        } else {
            return ParseText(filePath);
        }
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to parse class file: " << e.what() << std::endl;
        return names;
    }
}

std::vector<std::string> ClassNamesReader::ToVector(const std::map<int, std::string>& map) {
    if (map.empty()) return {};
    
    int maxId = map.rbegin()->first;
    // Pre-fill with "Unknown" to handle sparse IDs
    std::vector<std::string> vec(maxId + 1, "Unknown");
    
    for (const auto& [id, name] : map) {
        vec[id] = name;
    }
    return vec;
}

std::map<int, std::string> ClassNamesReader::ParseText(const std::string& path) {
    std::map<int, std::string> names;
    std::ifstream file(path);
    std::string line;
    int id = 0;
    
    while (std::getline(file, line)) {
        // Manual Trim
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        size_t last = line.find_last_not_of(" \t\r\n");
        
        std::string name = line.substr(first, (last - first + 1));
        names[id++] = name;
    }
    return names;
}

std::map<int, std::string> ClassNamesReader::ParseYaml(const std::string& path) {
    std::map<int, std::string> names;
    YAML::Node config = YAML::LoadFile(path);

    if (!config["names"]) {
        std::cerr << "[Warning] YAML file missing 'names' key." << std::endl;
        return names;
    }

    const YAML::Node& namesNode = config["names"];

    // Case 1: List Format (Your provided file)
    // names: ['Maize', 'Sugar beet', ...]
    if (namesNode.IsSequence()) {
        int id = 0;
        for (const auto& item : namesNode) {
            names[id++] = item.as<std::string>();
        }
    }
    // Case 2: Dictionary Format
    // names: {0: 'Maize', 1: 'Sugar beet'}
    else if (namesNode.IsMap()) {
        for (const auto& item : namesNode) {
            int id = item.first.as<int>();
            names[id] = item.second.as<std::string>();
        }
    }

    return names;
}

} // namespace cropandweed