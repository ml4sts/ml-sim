from setuptools import setup

setup(name='fairsim',
      version='0.1',
      description='synthetic data genertors for fairness assessment',
      url='http://github.com/brownsarahm/fair-sim/',
      author='Sarah M Brown',
      author_email='smb@sarahmbrown.org',
      license='MIT',
      packages=['fairsim'],
      zip_safe=False,
      install_requires=['Numpy', 'Scipy'])
