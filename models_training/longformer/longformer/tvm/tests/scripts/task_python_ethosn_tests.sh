#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
source tests/scripts/setup-pytest-env.sh


# Rebuild cython
# TODO(u99127): Enable cython tests.

find . -type f -path "*.pyc" | xargs rm -f
make cython3

# Note: Default behaviour is to assume the test target is Ethos-N78
# but setting ETHOSN_VARIANT_CONFIG appropriately
# (e.g. ETHOSN_VARIANT_CONFIG=Ethos-N78_1TOPS_2PLE_RATIO)
# switches the target to various Ethos-N78 configurations.
run_pytest ctypes python-ethosn tests/python/contrib/test_ethosn
