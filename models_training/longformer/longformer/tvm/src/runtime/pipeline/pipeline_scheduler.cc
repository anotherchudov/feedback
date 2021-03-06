/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "pipeline_scheduler.h"

#include <utility>
#include <vector>
namespace tvm {
namespace runtime {
/*!
 * \brief Initialize the pipeline.
 * \param modules The list of graph executor modules.
 * \param pipeline_conf The dependency information of each graph executor module.
 */
std::vector<std::shared_ptr<BackendRuntime>> PipelineScheduler::PipelineInit(
    const std::vector<Module>& modules, const ConfigPipelineExecution& pipeline_config) {
  std::vector<std::shared_ptr<BackendRuntime>> runtimes;
  graph_modules_ = modules;
  for (size_t i = 0; i < graph_modules_.size(); i++) {
    auto runItem = std::make_shared<BackendRuntime>(graph_modules_[i], i);
    runtimes.push_back(runItem);
  }
  return runtimes;
}
}  // namespace runtime
}  // namespace tvm
