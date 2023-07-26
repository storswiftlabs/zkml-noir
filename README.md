# zkML-Noir



## Description

Python implement ML model transcoding Noir contracts

## Directory structure

- tests: Testcase for transpiler
- transpiler: Python2Noir transpiler, include Noir language diffent module
  - context: Process the transformation context to build the complete Noir file
  - core_module: Include some core components such as struct and function
  - others_module: Include non-core statements
  - sub_module: Include some base components such as primitive type sign and key words
  - util: tools such as code format and logger 