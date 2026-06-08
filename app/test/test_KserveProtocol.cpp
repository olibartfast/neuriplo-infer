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

// --- Binary Tensor Data Extension -------------------------------------------

TEST(KserveProtocol, ParseHttpResponseInferenceHeaderLength) {
  const std::string raw = "HTTP/1.1 200 OK\r\n"
                          "Content-Length: 20\r\n"
                          "Inference-Header-Content-Length: 12\r\n\r\n"
                          "{\"outputs\":1}ABCDEFG";
  const auto resp = kserve::parseHttpResponse(raw);
  EXPECT_EQ(resp.status, 200);
  EXPECT_EQ(resp.inference_header_length, 12);
  // Plain JSON responses leave the field at -1.
  const std::string plain = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
  EXPECT_EQ(kserve::parseHttpResponse(plain).inference_header_length, -1);
}

TEST(KserveProtocol, AppendBinaryInputBuildsNodeAndBlob) {
  const float values[] = {1.5F, -2.0F};
  const auto bytes = bytesOf(values, sizeof(values));
  std::vector<std::uint8_t> blob;
  const auto node =
      kserve::appendBinaryInput("input0", "FP32", {1, 2}, bytes, blob);
  EXPECT_EQ(node["name"], "input0");
  EXPECT_EQ(node["datatype"], "FP32");
  EXPECT_FALSE(node.contains("data")); // binary inputs omit inline data
  EXPECT_EQ(node["parameters"]["binary_data_size"].get<std::size_t>(),
            bytes.size());
  EXPECT_EQ(blob, bytes);

  // A second input appends to the same blob in order.
  const std::uint8_t more[] = {0xAA, 0xBB};
  const auto more_bytes = bytesOf(more, sizeof(more));
  kserve::appendBinaryInput("input1", "UINT8", {2}, more_bytes, blob);
  ASSERT_EQ(blob.size(), bytes.size() + more_bytes.size());
  EXPECT_EQ(blob[bytes.size()], 0xAA);
  EXPECT_EQ(blob[bytes.size() + 1], 0xBB);
}

TEST(KserveProtocol, RequestBinaryOutputAndBinaryDataSize) {
  const auto out = kserve::requestBinaryOutput("logits");
  EXPECT_EQ(out["name"], "logits");
  EXPECT_TRUE(out["parameters"]["binary_data"].get<bool>());

  nlohmann::json with_size;
  with_size["parameters"]["binary_data_size"] = 16;
  EXPECT_EQ(kserve::binaryDataSize(with_size), 16);

  nlohmann::json without; // inline JSON tensor
  without["data"] = nlohmann::json::array();
  EXPECT_EQ(kserve::binaryDataSize(without), -1);
}

TEST(KserveProtocol, SplitBinaryBodyAndSlice) {
  nlohmann::json header;
  header["outputs"] = nlohmann::json::array();
  const std::string header_str = header.dump();
  const std::vector<std::uint8_t> tensor = {0x01, 0x02, 0x03, 0x04};
  std::string body = header_str;
  body.append(reinterpret_cast<const char *>(tensor.data()), tensor.size());

  std::vector<std::uint8_t> blob;
  const auto parsed = kserve::splitBinaryBody(body, header_str.size(), blob);
  EXPECT_TRUE(parsed.contains("outputs"));
  EXPECT_EQ(blob, tensor);

  EXPECT_EQ(kserve::sliceBlob(blob, 0, 2),
            (std::vector<std::uint8_t>{0x01, 0x02}));
  EXPECT_EQ(kserve::sliceBlob(blob, 2, 2),
            (std::vector<std::uint8_t>{0x03, 0x04}));
}

TEST(KserveProtocol, SplitAndSliceRejectTruncation) {
  std::vector<std::uint8_t> blob;
  EXPECT_THROW(kserve::splitBinaryBody("{}", 100, blob), std::runtime_error);
  const std::vector<std::uint8_t> small = {0x00, 0x01};
  EXPECT_THROW(kserve::sliceBlob(small, 1, 5), std::runtime_error);
  EXPECT_THROW(kserve::sliceBlob(small, 3, 1), std::runtime_error);
}

#endif // NEURIPLO_INFER_WITH_KSERVE
