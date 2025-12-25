#ifndef LOG_HPP
#define LOG_HPP

#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <sstream>


namespace _log {

enum Level { DEBUG, INFO, WARN, ERROR, FATAL };

inline int get_verbosity() {
  static int verbosity = 3;
  return verbosity;
}

inline void log_message(Level lvl, const char* fmt, ...) {
  if (get_verbosity() < 0) return; // optional filter by level
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
}

inline void log_fatal(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  std::exit(EXIT_FAILURE);
}

std::string format_init_list(std::initializer_list<size_t> list) {
  std::ostringstream oss;
  oss << "(";
  for (auto it = list.begin(); it != list.end(); ++it) {
    oss << *it;
    if (std::next(it) != list.end()) oss << ", ";
  }
  oss << ")";
  return oss.str();
}

} // namespace _log

#endif // LOG_HPP
