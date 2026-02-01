#pragma once
#include <map>
#include <vector>
#include <string>

namespace cropandweed {

class ClassNamesReader {
public:
    /**
     * @brief Reads class names from a file (TXT or YAML).
     * @param filePath Path to the configuration file.
     * @return Map of ID -> ClassName.
     */
    static std::map<int, std::string> Read(const std::string& filePath);

    /**
     * @brief Converts the map to a vector, filling gaps with "Unknown".
     * Useful for passing data to Sink or Detector properties.
     */
    static std::vector<std::string> ToVector(const std::map<int, std::string>& map);

private:
    static std::map<int, std::string> ParseText(const std::string& path);
    
    // Implementation now hidden in .cpp, avoiding <yaml-cpp/yaml.h> leak
    static std::map<int, std::string> ParseYaml(const std::string& path);
};

} // namespace cropandweed