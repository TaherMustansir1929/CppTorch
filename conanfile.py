from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.build import check_min_cppstd
from conan.tools.files import copy
import os


class CppTorchConan(ConanFile):
    name = "cpptorch"
    version = "0.1.0"
    license = "MIT"
    url = "https://github.com/TaherMustansir1929/CppTorch"
    homepage = "https://github.com/TaherMustansir1929/CppTorch"
    description = "Minimal neural networks library with numpy-style array primitives"
    topics = ("neural-network", "deep-learning", "numpy", "blas")

    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    exports_sources = (
        "CMakeLists.txt",
        "cmake/*",
        "include/*",
        "LICENSE",
        "README.md",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def validate(self):
        check_min_cppstd(self, 20)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["cpptorch_BUILD_TESTS"] = False
        tc.variables["cpptorch_BUILD_EXAMPLES"] = False
        tc.variables["cpptorch_INSTALL"] = True
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "cpptorch")
        self.cpp_info.set_property("cmake_target_name", "CppTorch::cpptorch")
        self.cpp_info.bindirs = ["lib"]
        self.cpp_info.libdirs = ["lib"]
