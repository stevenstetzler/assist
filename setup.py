import inspect
import os
import sys
import urllib.request
from codecs import open
from distutils import sysconfig
from distutils.sysconfig import get_python_lib

try:
    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.install import install as _install
except ImportError:
    print("Installing ASSIST requires setuptools.  Do 'pip install setuptools'.")
    sys.exit(1)

# URLs for the required BSP ephemeris files
_BSP_FILES = {
    "de440.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp",
    "sb441-n16.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp",
}

_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def download_bsp_files(data_dir=None):
    """Download the required BSP ephemeris files into *data_dir* (default: ./data)."""
    if data_dir is None:
        data_dir = _DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    for filename, url in _BSP_FILES.items():
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            print(f"  {filename} already present, skipping.")
            continue
        print(f"  Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")


class download_data(_install):
    """Custom 'setup.py download_data' command to fetch BSP ephemeris files."""

    description = "Download required BSP ephemeris files into the data/ directory"
    user_options = _install.user_options + [
        ("data-dir=", None, "Directory in which to place the downloaded BSP files"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.data_dir = None

    def finalize_options(self):
        super().finalize_options()
        if self.data_dir is None:
            self.data_dir = _DATA_DIR

    def run(self):
        download_bsp_files(self.data_dir)

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# Try to get git hash
try:
    import subprocess
    # `git rev-parse HEAD` includes a trailing newline; strip it so the
    # preprocessor macro doesn't embed a newline (which can confuse compilers).
    ghash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    ghash_arg = "-DASSISTGITHASH="+ghash
except:
    ghash_arg = "-DASSISTGITHASH=aa1cbe02ba7396da94f3f5154012b0c63e1ec2ac" #GITHASHAUTOUPDATE

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        try:
            import rebound
        except ImportError:
            print("ASSIST did not automatically install REBOUND.  Please try first installing REBOUND (https://rebound.readthedocs.org/en/latest/python_quickstart.html")
            sys.exit(1)
        try:
            version = rebound.__version__ # Added in 2.12.1
        except AttributeError:
            print("ASSIST did not automatically install a recent enough version of REBOUND.  Try upgrading REBOUND.  See 5.3 in https://rebound.readthedocs.org/en/latest/python_quickstart.html")
            sys.exit(1)

        rebdir = os.path.dirname(inspect.getfile(rebound))
        # get site-packages dir to add to paths in case REBOUND & ASSIST installed simul in tmp dir
        rebdirsp = get_python_lib()+'/'#[p for p in sys.path if p.endswith('site-packages')][0]+'/'
        self.include_dirs.append(rebdir)
        # Keep this list in sync with libassistmodule.sources below.
        sources = ['src/assist.c', 'src/spk.c', 'src/forces.c', 'src/tools.c', 'src/ascii_ephem.c']

        if not "CONDA_BUILD_CROSS_COMPILATION" in os.environ:
            # Add library directories for build-time linking
            self.library_dirs.append(rebdir+'/../')
            self.library_dirs.append(rebdirsp)
            
            for ext in self.extensions:
                # Clear any existing runtime_library_dirs to avoid conflicts
                ext.runtime_library_dirs = []
                
                # Use loader-relative paths for runtime linking
                if sys.platform == 'darwin':
                    # Primary path: same directory as the loading library
                    ext.extra_link_args.append('-Wl,-rpath,@loader_path')
                    # Backup path: current directory (for compatibility)
                    ext.extra_link_args.append('-Wl,-rpath,@loader_path/.')
                elif sys.platform.startswith('linux'):
                    # Linux equivalent of @loader_path
                    ext.extra_link_args.append('-Wl,-rpath,$ORIGIN')
                    ext.extra_link_args.append('-Wl,-rpath,$ORIGIN/.')
                
                print("Library directories:", self.library_dirs)
                print("Extra link args:", ext.extra_link_args)
        else:
            # For conda-forge cross-compile builds
            rebdir=get_python_lib(prefix=os.environ["PREFIX"])
            self.library_dirs.append(rebdir)
            for ext in self.extensions:
                ext.extra_link_args.append('-Wl,-rpath,'+rebdir)
                print("Cross-compile extra link args:", ext.extra_link_args)

from distutils.version import LooseVersion

extra_link_args=[]
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-shared')
    extra_link_args.append('-Wl,-install_name,@rpath/libassist'+suffix)
elif sys.platform.startswith('linux'):
    # Linux doesn't need install_name, but we can add other flags if needed
    pass

libassistmodule = Extension(
    'libassist',
    sources=[
        'src/assist.c',
        'src/spk.c',
        'src/forces.c',
        'src/tools.c',
        'src/ascii_ephem.c',
    ],
    include_dirs=['src'],
    library_dirs=[],
    runtime_library_dirs=[],  # Will be set by build_ext.finalize_options
    libraries=['rebound' + suffix[: suffix.rfind('.')]],
    define_macros=[('LIBASSIST', None)],
    extra_compile_args=[
        # `spk.c` reads memory-mapped binary data via typed pointers. This is
        # common practice, but it can become undefined behavior under aggressive
        # strict-aliasing optimizations, and we have observed that it can change
        # long integrations (e.g. Apophis) by O(100 m).
        #
        # Compile ASSIST with strict-aliasing disabled to keep the Python build
        # numerically consistent with the `src/` Makefile build and with the C
        # unit tests.
        '-fno-strict-aliasing',
        # Distutils injects `-fno-strict-overflow` on some platforms (notably
        # macOS). We have observed that this alone can change sensitive long
        # integrations (e.g. Apophis) by O(100 m) because it changes compiler
        # optimizations and therefore floating-point roundoff / adaptive step
        # choices.
        #
        # The `src/Makefile` build (and C unit tests) rely on the compiler
        # default here, which is effectively "strict overflow" at -O3. Add the
        # explicit override so the Python extension matches the Makefile build.
        #
        # Note: This flag is not supported by MSVC, so we only apply it on
        # non-Windows platforms.
        *([] if sys.platform.startswith('win') else ['-fstrict-overflow']),
        '-O3',
        '-std=c99',
        '-fPIC',
        '-D_GNU_SOURCE',
        '-Wpointer-arith',
        ghash_arg,
    ],
    extra_link_args=extra_link_args,
)

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='assist',
    version='1.1.9',
    description='A library high accuracy ephemeris in REBOUND',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/matthewholman/assist',
    author='Matthew Holman',
    author_email='mholman@cfa.harvard.edu',
    license='GPL',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy astrophysics nbody integrator',
    packages=['assist'],
    package_data={"assist": ["assist.h", "py.typed"]},
    cmdclass={'build_ext': build_ext, 'download_data': download_data},
    setup_requires=['rebound>=4.4.11', 'numpy'],
    install_requires=['rebound>=4.4.11', 'numpy'],
    tests_require=["numpy","matplotlib","rebound"],
    test_suite="assist.test",
    ext_modules = [libassistmodule],
    zip_safe=False) 