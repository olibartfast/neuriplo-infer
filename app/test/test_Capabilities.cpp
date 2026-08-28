#include "Capabilities.hpp"
#include "RunReport.hpp"
#include "TaskRouting.hpp"

#include "neuriplo/tasks/core/result_types.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using json = nlohmann::json;

const json &findById(const json &items, const std::string &id) {
  const auto found =
      std::find_if(items.begin(), items.end(),
                   [&id](const json &item) { return item.at("id") == id; });
  if (found == items.end()) {
    throw std::runtime_error("Capability not found: " + id);
  }
  return *found;
}

bool containsString(const json &values, const std::string &expected) {
  return std::find(values.begin(), values.end(), expected) != values.end();
}

neuriplo_tasks::TaskType taskTypeForId(const std::string &id) {
  using neuriplo_tasks::TaskType;
  if (id == "object_detection") {
    return TaskType::Detection;
  }
  if (id == "instance_segmentation") {
    return TaskType::InstanceSegmentation;
  }
  if (id == "classification") {
    return TaskType::Classification;
  }
  if (id == "video_classification") {
    return TaskType::VideoClassification;
  }
  if (id == "optical_flow") {
    return TaskType::OpticalFlow;
  }
  if (id == "pose_estimation") {
    return TaskType::PoseEstimation;
  }
  if (id == "depth_estimation") {
    return TaskType::DepthEstimation;
  }
  if (id == "open_vocabulary_detection") {
    return TaskType::OpenVocabDetection;
  }
  if (id == "image_understanding") {
    return TaskType::ImageUnderstanding;
  }
  if (id == "gaussian_splatting") {
    return TaskType::GaussianSplatting;
  }
  throw std::runtime_error("Unexpected task capability: " + id);
}
} // namespace

TEST(CapabilitiesContract, HasStableTopLevelShape) {
  const json capabilities = buildCapabilities();

  EXPECT_EQ(capabilities.at("schema_version"), 1);
  EXPECT_EQ(capabilities.at("producer").at("name"), "neuriplo-infer");
  EXPECT_TRUE(capabilities.at("producer").at("version").is_string());
  EXPECT_TRUE(capabilities.at("execution").at("workflows").is_array());
  EXPECT_TRUE(capabilities.at("tasks").is_array());
  EXPECT_TRUE(capabilities.at("parameters").is_object());
  EXPECT_TRUE(capabilities.at("source_types").is_array());
}

TEST(CapabilitiesContract, AdvertisesWhereRunDiagnosticsAreWritten) {
  const json run_report =
      buildCapabilities().at("diagnostics").at("run_report");

  // A consumer must be able to find the document without knowing its name, and
  // must be able to tell which stage vocabulary it will see.
  EXPECT_EQ(run_report.at("schema_version"),
            neuriplo_infer::RunReport::kSchemaVersion);
  EXPECT_EQ(run_report.at("path"), neuriplo_infer::RunReport::kDefaultPath);

  const json stages = run_report.at("stages");
  ASSERT_TRUE(stages.is_array());
  for (const auto stage :
       {neuriplo_infer::RunStage::Configuration,
        neuriplo_infer::RunStage::ModelLoad, neuriplo_infer::RunStage::Source,
        neuriplo_infer::RunStage::Preprocess,
        neuriplo_infer::RunStage::Inference,
        neuriplo_infer::RunStage::Postprocess,
        neuriplo_infer::RunStage::Render, neuriplo_infer::RunStage::Unknown}) {
    EXPECT_TRUE(containsString(stages, std::string(toString(stage))))
        << toString(stage);
  }
  EXPECT_EQ(stages.size(), 8u);
}

TEST(CapabilitiesContract, AdvertisedModelSelectorsMatchAppRouting) {
  const json tasks = buildCapabilities().at("tasks");

  for (const auto &task : tasks) {
    const auto expected_type = taskTypeForId(task.at("id"));
    for (const auto &model : task.at("models")) {
      EXPECT_EQ(getTaskTypeForModel(model.at("id")), expected_type)
          << model.at("id");
      for (const auto &alias : model.at("aliases")) {
        EXPECT_EQ(getTaskTypeForModel(alias), expected_type) << alias;
      }
    }
  }
}

TEST(CapabilitiesContract, AdvertisesRoutedAliasesAndFamilies) {
  const json tasks = buildCapabilities().at("tasks");

  const json &detection = findById(tasks, "object_detection");
  const json &rtdetr = findById(detection.at("models"), "rtdetr");
  for (const char *alias : {"rtdetrv2", "dfine", "deim", "deimv2"}) {
    EXPECT_TRUE(containsString(rtdetr.at("aliases"), alias)) << alias;
  }
  EXPECT_TRUE(containsString(rtdetr.at("patterns"), "deim*"));

  const json &edgecrafter = findById(detection.at("models"), "ecdet");
  EXPECT_TRUE(containsString(edgecrafter.at("patterns"), "edgecrafter*det*"));

  const json &pose = findById(tasks, "pose_estimation");
  const json &yolo_pose = findById(pose.at("models"), "yolov8pose");
  EXPECT_TRUE(containsString(yolo_pose.at("patterns"), "yolo*pose*"));
  const json &rfdetr_pose = findById(pose.at("models"), "rfdetrpose");
  EXPECT_TRUE(containsString(rfdetr_pose.at("patterns"), "rfdetr*pose*"));
}

TEST(CapabilitiesContract, ParameterReferencesResolve) {
  const json capabilities = buildCapabilities();
  const json &catalog = capabilities.at("parameters");

  const auto expect_references_resolve = [&catalog](const json &selection) {
    for (const char *group : {"required", "optional"}) {
      for (const auto &parameter : selection.at(group)) {
        EXPECT_TRUE(catalog.contains(parameter.get<std::string>()))
            << parameter;
      }
    }
  };

  for (const auto &workflow : capabilities.at("execution").at("workflows")) {
    expect_references_resolve(workflow.at("parameters"));
  }
  for (const auto &task : capabilities.at("tasks")) {
    expect_references_resolve(task.at("parameters"));
    for (const auto &model : task.at("models")) {
      expect_references_resolve(model.at("parameters"));
    }
  }
}

TEST(CapabilitiesContract, IdentifiersAreUniqueWithinTheirScopes) {
  const json capabilities = buildCapabilities();

  std::set<std::string> task_ids;
  for (const auto &task : capabilities.at("tasks")) {
    EXPECT_TRUE(task_ids.insert(task.at("id").get<std::string>()).second)
        << task.at("id");

    std::set<std::string> selectors;
    for (const auto &model : task.at("models")) {
      EXPECT_TRUE(selectors.insert(model.at("id").get<std::string>()).second)
          << model.at("id");
      for (const auto &alias : model.at("aliases")) {
        EXPECT_TRUE(selectors.insert(alias.get<std::string>()).second) << alias;
      }
    }
  }
}

TEST(CapabilitiesContract, EnumDefaultsAreAdvertisedValues) {
  for (const auto &[id, parameter] :
       buildCapabilities().at("parameters").items()) {
    if (parameter.at("value_type") != "enum" ||
        !parameter.contains("default")) {
      continue;
    }

    EXPECT_NE(std::find(parameter.at("values").begin(),
                        parameter.at("values").end(), parameter.at("default")),
              parameter.at("values").end())
        << id;
  }
}

TEST(CapabilitiesContract, ReportsOnlyCompiledExecutionWorkflows) {
  const json workflows = buildCapabilities().at("execution").at("workflows");

#ifdef NEURIPLO_INFER_WITH_LOCAL_BACKENDS
  const json &local = findById(workflows, "local");
  ASSERT_EQ(local.at("backends").size(), 1U);
  EXPECT_FALSE(local.at("backends").at(0).get<std::string>().empty());
#else
  EXPECT_THROW(findById(workflows, "local"), std::runtime_error);
#endif

#ifdef NEURIPLO_INFER_WITH_KSERVE
  const json &remote = findById(workflows, "client_server");
  const json &protocol = findById(remote.at("protocols"), "kserve_v2");
  EXPECT_EQ(
      protocol.at("transports"),
      buildCapabilities().at("parameters").at("kserve_transport").at("values"));
  EXPECT_NE(std::find(protocol.at("transports").begin(),
                      protocol.at("transports").end(), "http"),
            protocol.at("transports").end());
#ifdef KSERVE_CLIENT_WITH_GRPC
  EXPECT_NE(std::find(protocol.at("transports").begin(),
                      protocol.at("transports").end(), "grpc"),
            protocol.at("transports").end());
#endif

  const auto &parameters = remote.at("parameters").at("optional");
  for (const char *parameter :
       {"input_mode", "task_model", "task_model_version", "postprocess_mode"}) {
    EXPECT_NE(std::find(parameters.begin(), parameters.end(), parameter),
              parameters.end())
        << parameter;
  }
#else
  EXPECT_THROW(findById(workflows, "client_server"), std::runtime_error);
#endif
}
