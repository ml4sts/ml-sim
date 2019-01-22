from setuptools import setup

setup(name='mlsim',
      version='0.1',
      description='synthetic data generators for machine learning evaluation',
      url='http://github.com/brownsarahm/ml-sim/',
      author='Sarah M Brown',
      author_email='smb@sarahmbrown.org',
      license='MIT',
      packages=['mlsim'],
      zip_safe=False,
      install_requires=['Numpy', 'Scipy'])
