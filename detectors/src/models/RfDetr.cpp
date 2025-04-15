#include "RfDetr.hpp"

RfDetr::RfDetr(const ModelInfo& model_info, float confidenceThreshold) : Detector{model_info, confidenceThreshold}
{
    std::unordered_map<std::string, size_t> output_name_to_index;
    const auto& outputs = model_info.getOutputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        output_name_to_index[outputs[i].name] = i;
    }

    if (output_name_to_index.count("dets")) dets_idx_ = output_name_to_index["dets"];
    if (output_name_to_index.count("labels")) labels_idx_ = output_name_to_index["labels"];

    // Check if all indices are set
    if (!dets_idx_.has_value() || !labels_idx_.has_value()) {
        throw std::runtime_error("Not all required output indices were set in the model info");
    }
}
std::vector<Detection> RfDetr::postprocess(
    const std::vector<std::vector<TensorElement>>& outputs,
    const std::vector<std::vector<int64_t>>& shapes,
    const cv::Size& frame_size)
{
    const auto& boxes = outputs[dets_idx_.value()];
    const std::vector<int64_t>& shape_boxes = shapes[dets_idx_.value()];
    const auto& labels = outputs[labels_idx_.value()];
    const std::vector<int64_t>& shape_labels = shapes[labels_idx_.value()];

    std::vector<Detection> detections;

    if (shape_boxes.size() < 3 || shape_labels.size() < 3) {
        throw std::runtime_error("Invalid output tensor shapes");
    }

    const size_t num_detections = shape_boxes[1];
    const size_t num_classes = shape_labels[2];

    const float scale_w = static_cast<float>(frame_size.width) / network_width_;
    const float scale_h = static_cast<float>(frame_size.height) / network_height_;

    for (size_t i = 0; i < num_detections; ++i) {
        const size_t det_offset = i * shape_boxes[2];
        const size_t label_offset = i * num_classes;

        float max_score = -1.0f;
        int max_class_idx = -1;
        for (size_t j = 0; j < num_classes; ++j) {
            float logit;
            try {
                logit = std::get<float>(labels[label_offset + j]);
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("Invalid TensorElement type for labels at index " + std::to_string(label_offset + j));
            }
            const float score = sigmoid(logit);
            if (score > max_score) {
                max_score = score;
                max_class_idx = j;
            }
        }

        max_class_idx -= 1;

        if (max_score > confidenceThreshold_ && max_class_idx >= 0 &&
            static_cast<size_t>(max_class_idx) < num_classes) {
            float x_center, y_center, width, height;
            try {
                x_center = std::get<float>(boxes[det_offset + 0]) * network_width_;
                y_center = std::get<float>(boxes[det_offset + 1]) * network_height_;
                width = std::get<float>(boxes[det_offset + 2]) * network_width_;
                height = std::get<float>(boxes[det_offset + 3]) * network_height_;
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("Invalid TensorElement type for boxes at index " + std::to_string(det_offset));
            }

            const float x_min = x_center - width / 2.0f;
            const float y_min = y_center - height / 2.0f;
            const float x_max = x_center + width / 2.0f;
            const float y_max = y_center + height / 2.0f;

            cv::Rect bbox(
                static_cast<int>(x_min * scale_w),
                static_cast<int>(y_min * scale_h),
                static_cast<int>((x_max - x_min) * scale_w),
                static_cast<int>((y_max - y_min) * scale_h)
            );

            Detection detection;
            detection.bbox = bbox;
            detection.score = max_score;
            detection.label = max_class_idx;

            detections.push_back(detection);
        }
    }

    return detections;
}

cv::Mat RfDetr::preprocess_image(const cv::Mat& image) {
    cv::Mat output_image;
    cv::cvtColor(image, output_image, cv::COLOR_BGR2RGB);
    cv::resize(output_image, output_image, cv::Size(network_width_, network_height_));

    // Apply scaling and normalization as per pseudocode
    cv::Mat flt_image;
    output_image.convertTo(flt_image, CV_32FC3, 1.f / 255.f); // Scale to 0-1
    cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image); // Mean subtraction
    cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image); // Std division

    return flt_image;
}
