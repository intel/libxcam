/*
 * x3a_analyzer_loader.h - x3a analyzer loader
 *
 *  Copyright (c) 2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_3A_ANALYZER_LOADER_H
#define XCAM_3A_ANALYZER_LOADER_H

#include <base/xcam_common.h>
#include <base/xcam_3a_description.h>
#include "analyzer_loader.h"
#include "smartptr.h"

namespace XCam {
class IspController;
class HybridAnalyzer;
class X3aAnalyzer;

class X3aAnalyzerLoader
    : public AnalyzerLoader
{
public:
    X3aAnalyzerLoader (const char *lib_path, const char *symbol = XCAM_3A_LIB_DESCRIPTION);
    virtual ~X3aAnalyzerLoader ();

    SmartPtr<X3aAnalyzer> load_dynamic_analyzer (SmartPtr<X3aAnalyzerLoader> &self);
    SmartPtr<X3aAnalyzer> load_hybrid_analyzer (SmartPtr<X3aAnalyzerLoader> &self,
            SmartPtr<IspController> &isp,
            const char *cpf_path);

protected:
    virtual void *load_symbol (void* handle);

private:
    XCAM_DEAD_COPY(X3aAnalyzerLoader);
};

};

#endif //XCAM_3A_ANALYZER_LOADER_H
