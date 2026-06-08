#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "KserveProtocol.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

std::vector<std::uint8_t> bytesOf(const void *p, std::size_t n) {
  std::vector<std::uint8_t> out(n);
  std::memcpy(out.data(), p, n);
  return out;
}

} // namespace

TEST(KserveProtocol, ParseEndpointHostPortPrefix) {
  const auto ep =
      kserve::parseEndpoint("http://127.0.0.1:19090/gateway/", 8080);
  EXPECT_EQ(ep.host, "127.0.0.1");
  EXPECT_EQ(ep.port, 19090);
  EXPECT_EQ(ep.path_prefix, "/gateway"); // trailing slash trimmed
  EXPECT_FALSE(ep.tls);
}

TEST(KserveProtocol, ParseEndpointDefaultsAndTls) {
  const auto plain = kserve::parseEndpoint("triton.local", 8001);
  EXPECT_EQ(plain.host, "triton.local");
  EXPECT_EQ(plain.port, 8001);
  EXPECT_TRUE(plain.path_prefix.empty());

  EXPECT_TRUE(kserve::parseEndpoint("https://h:443", 80).tls);
  EXPECT_TRUE(kserve::parseEndpoint("grpcs://h:9000", 80).tls);
  EXPECT_FALSE(kserve::parseEndpoint("grpc://h:9000", 80).tls);
}

TEST(KserveProtocol, ParseHttpResponseContentLength) {
  const std::string raw =
      "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"ok\":true}\r\n";
  const auto resp = kserve::parseHttpResponse(raw);
  EXPECT_EQ(resp.status, 200);
  EXPECT_EQ(resp.body, "{\"ok\":true}\r\n");
}

TEST(KserveProtocol, ParseHttpResponseChunked) {
  // "Wikipedia" classic chunked example, JSON-ish payload.
  const std::string raw = "HTTP/1.1 200 OK\r\n"
                          "Transfer-Encoding: chunked\r\n\r\n"
                          "4\r\n{\"a\"\r\n"
                          "3\r\n:1}\r\n"
                          "0\r\n\r\n";
  const auto resp = kserve::parseHttpResponse(raw);
  EXPECT_EQ(resp.status, 200);
  EXPECT_EQ(resp.body, "{\"a\":1}");
}

TEST(KserveProtocol, ParseHttpResponseError) {
  EXPECT_THROW(kserve::parseHttpResponse("garbage-with-no-status"),
               std::runtime_error);
}

TEST(KserveProtocol, HealthPaths) {
  EXPECT_EQ(kserve::serverLivePath(), "/v2/health/live");
  EXPECT_EQ(kserve::serverReadyPath(), "/v2/health/ready");
  EXPECT_EQ(kserve::modelReadyPath("resnet", "1"),
            "/v2/models/resnet/versions/1/ready");
  EXPECT_EQ(kserve::modelReadyPath("resnet", ""), "/v2/models/resnet/ready");
}

TEST(KserveProtocol, DatatypeByteWidth) {
  EXPECT_EQ(kserve::datatypeByteWidth("UINT8"), 1u);
  EXPECT_EQ(kserve::datatypeByteWidth("FP16"), 2u);
  EXPECT_EQ(kserve::datatypeByteWidth("FP32"), 4u);
  EXPECT_EQ(kserve::datatypeByteWidth("INT64"), 8u);
  EXPECT_EQ(kserve::datatypeByteWidth("BYTES"), 0u);
}

TEST(KserveProtocol, EncodeFp32) {
  const float values[] = {1.5F, -2.0F};
  const auto bytes = bytesOf(values, sizeof(values));
  const auto json = kserve::encodeTensorData(bytes, "FP32");
  ASSERT_EQ(json.size(), 2u);
  EXPECT_FLOAT_EQ(json[0].get<float>(), 1.5F);
  EXPECT_FLOAT_EQ(json[1].get<float>(), -2.0F);
}

TEST(KserveProtocol, EncodeUint8AndInt64) {
  const std::uint8_t u8[] = {0, 127, 255};
  const auto j8 = kserve::encodeTensorData(bytesOf(u8, sizeof(u8)), "UINT8");
  ASSERT_EQ(j8.size(), 3u);
  EXPECT_EQ(j8[2].get<unsigned>(), 255u);

  const std::int64_t i64[] = {-5, 1LL << 40};
  const auto j64 = kserve::encodeTensorData(bytesOf(i64, sizeof(i64)), "INT64");
  ASSERT_EQ(j64.size(), 2u);
  EXPECT_EQ(j64[1].get<std::int64_t>(), 1LL << 40);
}

TEST(KserveProtocol, EncodeRejectsMisalignedBytes) {
  const std::vector<std::uint8_t> bytes = {0x00, 0x01, 0x02};
  EXPECT_THROW(kserve::encodeTensorData(bytes, "FP32"), std::runtime_error);
}

TEST(KserveProtocol, EncodeRejectsUnknownDatatype) {
  const std::vector<std::uint8_t> bytes = {0x00};
  EXPECT_THROW(kserve::encodeTensorData(bytes, "BYTES"), std::runtime_error);
}

TEST(KserveProtocol, DecodeFp32RoundTrip) {
  const float values[] = {1.5F, -2.0F, 0.25F};
  const auto bytes = bytesOf(values, sizeof(values));
  const auto json = kserve::encodeTensorData(bytes, "FP32");
  const auto decoded = kserve::decodeTensorData(json, "FP32");
  EXPECT_EQ(decoded, bytes);
}

TEST(KserveProtocol, DecodeInt64AndUint8RoundTrip) {
  const std::int64_t i64[] = {-5, 1LL << 40};
  const auto b64 = bytesOf(i64, sizeof(i64));
  EXPECT_EQ(
      kserve::decodeTensorData(kserve::encodeTensorData(b64, "INT64"), "INT64"),
      b64);

  const std::uint8_t u8[] = {0, 127, 255};
  const auto b8 = bytesOf(u8, sizeof(u8));
  EXPECT_EQ(
      kserve::decodeTensorData(kserve::encodeTensorData(b8, "UINT8"), "UINT8"),
      b8);
}

TEST(KserveProtocol, DecodeRejectsNonArrayAndUnknown) {
  EXPECT_THROW(kserve::decodeTensorData(nlohmann::json::object(), "FP32"),
               std::runtime_error);
  EXPECT_THROW(kserve::decodeTensorData(nlohmann::json::array(), "BYTES"),
               std::runtime_error);
}

#endif // NEURIPLO_INFER_WITH_KSERVE
