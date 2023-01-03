# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools


with open("README.md", encoding="utf8") as f:
    readme = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name="paco",
        version="0.5",
        author="Meta Platforms, Inc.",
        description="OZI data scripts",
        long_description=readme,
        url="https://github.com/facebookresearch/paco.git",
        license="MIT",
        packages=setuptools.find_packages(),
        setup_requires=["pytest-runner"],
        test_requires=["pytest"],
    )
