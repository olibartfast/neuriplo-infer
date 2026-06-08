#include "KserveProtocol.hpp"

#include <cctype>
#include <cstring>
#include <stdexcept>

namespace kserve {

namespace {

std::string stripScheme(const std::string &url, bool &tls) {
  tls = false;
  if (url.rfind("https://", 0) == 0) {
    tls = true;
    return url.substr(8);
  }
  if (url.rfind("grpcs://", 0) == 0) {
    tls = true;
    return url.substr(8);
  }
  if (url.rfind("http://", 0) == 0) {
    return url.substr(7);
  }
  if (url.rfind("grpc://", 0) == 0) {
    return url.substr(7);
  }
  return url;
}

// Narrows a float to an IEEE-754 half-precision bit pattern (round toward
// zero).
std::uint16_t floatToHalf(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::int32_t exp =
      static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  const std::uint32_t mant = x & 0x7FFFFFu;
  if (exp >= 0x1F) {
    // overflow / inf / nan
    return static_cast<std::uint16_t>(sign | 0x7C00u |
                                      (mant != 0 ? 0x0200u : 0u));
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign; // underflow to zero
    }
    const std::uint32_t m =
        (mant | 0x800000u) >> static_cast<std::uint32_t>(14 - exp);
    return static_cast<std::uint16_t>(sign | m);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

// Reads a contiguous array of T from raw bytes and appends to a JSON array via
// the supplied projection (e.g. widening to a JSON-friendly type).
template <typename T, typename Proj>
nlohmann::json readArray(const std::vector<std::uint8_t> &bytes, Proj proj) {
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error("KServe input byte count is not a multiple of the "
                             "datatype width");
  }
  nlohmann::json data = nlohmann::json::array();
  const std::size_t count = bytes.size() / sizeof(T);
  for (std::size_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    data.push_back(proj(value));
  }
  return data;
}

// Reads a JSON numeric array element-by-element (via the supplied projection to
// T) and appends the raw little-endian bytes of each value.
template <typename T, typename Proj>
std::vector<std::uint8_t> writeArray(const nlohmann::json &data, Proj proj) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(data.size() * sizeof(T));
  for (const auto &elem : data) {
    const T value = proj(elem);
    const auto *p = reinterpret_cast<const std::uint8_t *>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  return bytes;
}

} // namespace

float halfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16;
  const std::uint32_t exp = (h & 0x7C00u) >> 10;
  const std::uint32_t mant = h & 0x03FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign; // +/- zero
    } else {
      // subnormal: normalise
      std::uint32_t e = 0;
      std::uint32_t m = mant;
      while ((m & 0x0400u) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x03FFu;
      bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13); // inf / nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

Endpoint parseEndpoint(const std::string &url, int default_port) {
  Endpoint ep;
  auto value = stripScheme(url, ep.tls);

  const auto slash = value.find('/');
  std::string authority = value;
  if (slash != std::string::npos) {
    authority = value.substr(0, slash);
    auto prefix = value.substr(slash);
    while (!prefix.empty() && prefix.back() == '/') {
      prefix.pop_back();
    }
    ep.path_prefix = prefix;
  }

  const auto colon = authority.find(':');
  if (colon != std::string::npos) {
    ep.host = authority.substr(0, colon);
    ep.port = std::stoi(authority.substr(colon + 1));
  } else {
    ep.host = authority;
    ep.port = default_port;
  }
  return ep;
}

HttpResponse parseHttpResponse(const std::string &raw) {
  HttpResponse response;

  const auto first_space = raw.find(' ');
  if (first_space == std::string::npos) {
    throw std::runtime_error("invalid HTTP response: no status line");
  }
  const auto second_space = raw.find(' ', first_space + 1);
  response.status =
      std::stoi(raw.substr(first_space + 1, second_space - first_space - 1));

  const auto header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    throw std::runtime_error("invalid HTTP response: no header terminator");
  }
  const std::string headers = raw.substr(0, header_end);
  std::string body = raw.substr(header_end + 4);

  // Case-insensitive scan for Transfer-Encoding: chunked.
  std::string lower;
  lower.reserve(headers.size());
  for (char c : headers) {
    lower.push_back(
        static_cast<char>(::tolower(static_cast<unsigned char>(c))));
  }
  const bool chunked = lower.find("transfer-encoding:") != std::string::npos &&
                       lower.find("chunked") != std::string::npos;

  // Binary Tensor Data Extension: capture the JSON-header length so callers can
  // split the body into JSON header + binary blob.
  const std::string marker = "inference-header-content-length:";
  const auto marker_pos = lower.find(marker);
  if (marker_pos != std::string::npos) {
    std::size_t v = marker_pos + marker.size();
    while (v < headers.size() && (headers[v] == ' ' || headers[v] == '\t')) {
      ++v;
    }
    try {
      response.inference_header_length =
          std::stol(headers.substr(v)); // stops at the trailing CRLF
    } catch (...) {
      response.inference_header_length = -1;
    }
  }

  if (!chunked) {
    response.body = std::move(body);
    return response;
  }

  // De-chunk: <hex-size>\r\n<data>\r\n ... 0\r\n\r\n
  std::string decoded;
  std::size_t pos = 0;
  while (pos < body.size()) {
    const auto line_end = body.find("\r\n", pos);
    if (line_end == std::string::npos) {
      break;
    }
    const std::string size_token = body.substr(pos, line_end - pos);
    const std::size_t semicolon = size_token.find(';'); // ignore extensions
    std::size_t chunk_size = 0;
    try {
      chunk_size = static_cast<std::size_t>(
          std::stoul(size_token.substr(0, semicolon), nullptr, 16));
    } catch (...) {
      break;
    }
    if (chunk_size == 0) {
      break;
    }
    const std::size_t data_start = line_end + 2;
    if (data_start + chunk_size > body.size()) {
      throw std::runtime_error("invalid chunked HTTP body: truncated chunk");
    }
    decoded.append(body, data_start, chunk_size);
    pos = data_start + chunk_size + 2; // skip trailing CRLF
  }
  response.body = std::move(decoded);
  return response;
}

std::string serverLivePath() { return "/v2/health/live"; }

std::string serverReadyPath() { return "/v2/health/ready"; }

std::string modelReadyPath(const std::string &model_name,
                           const std::string &model_version) {
  if (model_version.empty()) {
    return "/v2/models/" + model_name + "/ready";
  }
  return "/v2/models/" + model_name + "/versions/" + model_version + "/ready";
}

std::size_t datatypeByteWidth(const std::string &datatype) {
  if (datatype == "BOOL" || datatype == "INT8" || datatype == "UINT8") {
    return 1;
  }
  if (datatype == "FP16" || datatype == "BF16" || datatype == "INT16" ||
      datatype == "UINT16") {
    return 2;
  }
  if (datatype == "FP32" || datatype == "INT32" || datatype == "UINT32") {
    return 4;
  }
  if (datatype == "FP64" || datatype == "INT64" || datatype == "UINT64") {
    return 8;
  }
  return 0; // BYTES / unknown
}

nlohmann::json encodeTensorData(const std::vector<std::uint8_t> &bytes,
                                const std::string &datatype) {
  if (datatype == "FP32") {
    return readArray<float>(bytes, [](float v) { return v; });
  }
  if (datatype == "FP64") {
    return readArray<double>(bytes, [](double v) { return v; });
  }
  if (datatype == "FP16") {
    return readArray<std::uint16_t>(
        bytes, [](std::uint16_t v) { return halfToFloat(v); });
  }
  if (datatype == "INT8") {
    return readArray<std::int8_t>(
        bytes, [](std::int8_t v) { return static_cast<int>(v); });
  }
  if (datatype == "INT16") {
    return readArray<std::int16_t>(
        bytes, [](std::int16_t v) { return static_cast<int>(v); });
  }
  if (datatype == "INT32") {
    return readArray<std::int32_t>(bytes, [](std::int32_t v) { return v; });
  }
  if (datatype == "INT64") {
    return readArray<std::int64_t>(bytes, [](std::int64_t v) { return v; });
  }
  if (datatype == "UINT8" || datatype == "BOOL") {
    return readArray<std::uint8_t>(
        bytes, [](std::uint8_t v) { return static_cast<unsigned>(v); });
  }
  if (datatype == "UINT16") {
    return readArray<std::uint16_t>(
        bytes, [](std::uint16_t v) { return static_cast<unsigned>(v); });
  }
  if (datatype == "UINT32") {
    return readArray<std::uint32_t>(bytes, [](std::uint32_t v) { return v; });
  }
  if (datatype == "UINT64") {
    return readArray<std::uint64_t>(bytes, [](std::uint64_t v) { return v; });
  }
  throw std::runtime_error("unsupported KServe input datatype: " + datatype);
}

std::vector<std::uint8_t> decodeTensorData(const nlohmann::json &data,
                                           const std::string &datatype) {
  if (!data.is_array()) {
    throw std::runtime_error("KServe output data is not a JSON array");
  }
  if (datatype == "FP32") {
    return writeArray<float>(
        data, [](const nlohmann::json &e) { return e.get<float>(); });
  }
  if (datatype == "FP64") {
    return writeArray<double>(
        data, [](const nlohmann::json &e) { return e.get<double>(); });
  }
  if (datatype == "FP16") {
    return writeArray<std::uint16_t>(data, [](const nlohmann::json &e) {
      return floatToHalf(e.get<float>());
    });
  }
  if (datatype == "INT8") {
    return writeArray<std::int8_t>(data, [](const nlohmann::json &e) {
      return static_cast<std::int8_t>(e.get<int>());
    });
  }
  if (datatype == "INT16") {
    return writeArray<std::int16_t>(data, [](const nlohmann::json &e) {
      return static_cast<std::int16_t>(e.get<int>());
    });
  }
  if (datatype == "INT32") {
    return writeArray<std::int32_t>(
        data, [](const nlohmann::json &e) { return e.get<std::int32_t>(); });
  }
  if (datatype == "INT64") {
    return writeArray<std::int64_t>(
        data, [](const nlohmann::json &e) { return e.get<std::int64_t>(); });
  }
  if (datatype == "UINT8" || datatype == "BOOL") {
    return writeArray<std::uint8_t>(data, [](const nlohmann::json &e) {
      return static_cast<std::uint8_t>(e.get<unsigned>());
    });
  }
  if (datatype == "UINT16") {
    return writeArray<std::uint16_t>(data, [](const nlohmann::json &e) {
      return static_cast<std::uint16_t>(e.get<unsigned>());
    });
  }
  if (datatype == "UINT32") {
    return writeArray<std::uint32_t>(
        data, [](const nlohmann::json &e) { return e.get<std::uint32_t>(); });
  }
  if (datatype == "UINT64") {
    return writeArray<std::uint64_t>(
        data, [](const nlohmann::json &e) { return e.get<std::uint64_t>(); });
  }
  throw std::runtime_error("unsupported KServe output datatype: " + datatype);
}

nlohmann::json appendBinaryInput(const std::string &name,
                                 const std::string &datatype,
                                 const std::vector<int64_t> &shape,
                                 const std::vector<std::uint8_t> &bytes,
                                 std::vector<std::uint8_t> &blob) {
  nlohmann::json node;
  node["name"] = name;
  node["datatype"] = datatype;
  node["shape"] = shape;
  node["parameters"]["binary_data_size"] = bytes.size();
  blob.insert(blob.end(), bytes.begin(), bytes.end());
  return node;
}

nlohmann::json requestBinaryOutput(const std::string &name) {
  nlohmann::json node;
  node["name"] = name;
  node["parameters"]["binary_data"] = true;
  return node;
}

long binaryDataSize(const nlohmann::json &node) {
  if (!node.contains("parameters")) {
    return -1;
  }
  const auto &params = node["parameters"];
  if (!params.contains("binary_data_size")) {
    return -1;
  }
  return params["binary_data_size"].get<long>();
}

nlohmann::json splitBinaryBody(const std::string &body, std::size_t header_len,
                               std::vector<std::uint8_t> &blob) {
  if (header_len > body.size()) {
    throw std::runtime_error(
        "invalid binary response: header length exceeds body");
  }
  nlohmann::json header = nlohmann::json::parse(body.substr(0, header_len));
  blob.assign(body.begin() + static_cast<std::ptrdiff_t>(header_len),
              body.end());
  return header;
}

std::vector<std::uint8_t> sliceBlob(const std::vector<std::uint8_t> &blob,
                                    std::size_t offset, std::size_t size) {
  if (offset > blob.size() || size > blob.size() - offset) {
    throw std::runtime_error(
        "invalid binary response: tensor slice runs past the blob");
  }
  return std::vector<std::uint8_t>(
      blob.begin() + static_cast<std::ptrdiff_t>(offset),
      blob.begin() + static_cast<std::ptrdiff_t>(offset + size));
}

} // namespace kserve
