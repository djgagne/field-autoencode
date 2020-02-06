from setuptools import setup

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               ]

requires = ["numpy>=1.14",
            "pandas>=0.2",
            "scipy>=1.0",
            "matplotlib>=2.0",
            "xarray",
            "netcdf4",
            "tensorflow>=2.0"]

if __name__ == "__main__":
    setup(name="field-autoencode",
          version="0.1",
          description="Autoencoding for different types of spatial fields",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/djgagne/field-autoencode",
          packages=["fieldae"],
          scripts=[],
          data_files=[],
          keywords=["fields", "deep learning"],
          include_package_data=True,
          zip_safe=False,
          install_requires=requires
          )
