from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='sleep_rnn',
      version='0.1',
      description='RNN para determinar estado de sueño',
      long_description=readme(),
      url='https://github.com/rcasal/sleep_rnn.git',
      author='Ramiro Casal',
      author_email='rcasal@conicet.gov.ar',
      license='MIT',
      packages=['sleep_rnn'],
      entry_points={}
      install_requires=[
          'numpy',
          'scipy',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
