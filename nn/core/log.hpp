#pragma once
// ============================================================================
// NCA — Zero-Cost C++20 Structured Logger
// core/log.hpp
//
// Uses std::source_location (C++20) to capture file/line/function at the
// call site WITHOUT macros. The compiler resolves everything at compile time.
// ============================================================================

#include <iostream>
#include <string_view>
#include <source_location>

namespace nca::log {

enum class Level : uint8_t { DEBUG = 0, INFO = 1, WARN = 2, ERR = 3, FATAL = 4 };

// Global minimum level — anything below is compiled away by the optimizer.
inline Level g_min_level = Level::INFO;

inline constexpr const char* level_tag(Level l) {
    switch (l) {
        case Level::DEBUG: return "[DEBUG]";
        case Level::INFO:  return "[INFO ]";
        case Level::WARN:  return "[WARN ]";
        case Level::ERR:   return "[ERROR]";
        case Level::FATAL: return "[FATAL]";
    }
    return "[?????]";
}

// Core log function — std::source_location captures call site automatically.
inline void emit(
    Level level,
    std::string_view msg,
    const std::source_location loc = std::source_location::current()
) {
    if (level < g_min_level) return;
    // Extract just the filename from the full path (branchless find_last_of)
    std::string_view file = loc.file_name();
    auto pos = file.find_last_of("\\/");
    if (pos != std::string_view::npos) file = file.substr(pos + 1);

    std::cerr << level_tag(level) << " "
              << file << ":" << loc.line()
              << " (" << loc.function_name() << ") "
              << msg << "\n";
}

// Convenience wrappers
inline void debug(std::string_view msg, const std::source_location loc = std::source_location::current()) { emit(Level::DEBUG, msg, loc); }
inline void info (std::string_view msg, const std::source_location loc = std::source_location::current()) { emit(Level::INFO,  msg, loc); }
inline void warn (std::string_view msg, const std::source_location loc = std::source_location::current()) { emit(Level::WARN,  msg, loc); }
inline void error(std::string_view msg, const std::source_location loc = std::source_location::current()) { emit(Level::ERR,   msg, loc); }
inline void fatal(std::string_view msg, const std::source_location loc = std::source_location::current()) { emit(Level::FATAL, msg, loc); }

} // namespace nca::log
