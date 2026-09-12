#pragma once

// App-local copy of neuriplo's TensorDataType, used with the app-local
// InferenceMetadata when neuriplo-infer is built WITHOUT local backends
// (KServe-only). Keep the enumerators identical to neuriplo's
// backends/src/TensorDataType.hpp.

enum class TensorDataType { Float32, Int32, Int64, UInt8, Int8, Bool };
