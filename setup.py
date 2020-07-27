import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Environment-specific dependencies.
extras = {
    'mujoco': ['mujoco_py>=2.0', 'imageio'],
    #'bullet': ['pybullet>=1.7.8']
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


extensions = [
    Extension(
        "gym_brt.quanser.quanser_wrapper.quanser_wrapper",
        ["gym_brt/quanser/quanser_wrapper/quanser_wrapper.pyx"],
        include_dirs=["/opt/quanser/hil_sdk/include"],
        libraries=[
            "hil",
            "quanser_runtime",
            "quanser_common",
            "rt",
            "pthread",
            "dl",
            "m",
            "c",
        ],
        library_dirs=["/opt/quanser/hil_sdk/lib"],
    )
]

# If Cython is installed build from source otherwise use the precompiled version
try:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)
except ImportError:
    pass


# Hacky way to check if the HIL SDK is installed (allows to run on Mac OS)
is_hil_sdk_installed = False
if os.path.isdir("/opt/quanser/hil_sdk/lib"):
    is_hil_sdk_installed = True

setup(
    name="gym_brt",
    version=0.2,
    cmdclass={"build_ext": build_ext} if is_hil_sdk_installed else {},
    install_requires=[
        "numpy",
        "gym>=0.17",
        "matplotlib"
    ],
    setup_requires=["numpy"],
    extras_require=extras,
    ext_modules=extensions if is_hil_sdk_installed else None,
    description="Adapted version of Blue River's OpenAI Gym wrapper around Quanser hardware.",
    url="https://github.com/BlueRiverTech/quanser-openai-driver/",
    author="Blue River Technology, "
           "Intelligent Control Systems Group (Max Planck Institute for Intelligent Systems), "
           "Institute for Data Science in Mechanical Engineering (RWTH Aachen University)",
    license="MIT",
)
