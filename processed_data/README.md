This directory is used to store the processed data.The directory structure is as follows:
```shell
└─csn
   ├─java
   │  ├─test
   │  │  │  cleaned.json
   │  │  │  final.ast
   │  │  │  final.code
   │  │  │  final.comment
   │  │  │  final.func_name
   │  │  │  final.json
   │  │  │  sim.code
   │  │  │  sim.comment
   │  │  │  sim.func_name
   │  │  │  sim.json
   │  │  │  sim.pair.res
   │  │  │
   │  │  └─tmp
   │  │      │  code.token
   │  │      │  denoised.json
   │  │      │  sim.code.res
   │  │      │  sim.pair.res
   │  │      │  tmp.code
   │  │      │
   │  │      └─corpus
   │  │
   │  ├─train
   │  │  │  cleaned.json
   │  │  │  final.ast
   │  │  │  final.code
   │  │  │  final.comment
   │  │  │  final.func_name
   │  │  │  final.json
   │  │  │  sim.code
   │  │  │  sim.comment
   │  │  │  sim.func_name
   │  │  │  sim.json
   │  │  │  sim.pair.res
   │  │  │
   │  │  └─tmp
   │  │      │  code.token
   │  │      │  denoised.json
   │  │      │  sim.code.res
   │  │      │  sim.pair.res
   │  │      │  tmp.code
   │  │      │
   │  │      └─corpus
   │  │
   │  └─valid
   │      │  cleaned.json
   │      │  final.ast
   │      │  final.code
   │      │  final.comment
   │      │  final.func_name
   │      │  final.json
   │      │  sim.code
   │      │  sim.comment
   │      │  sim.func_name
   │      │  sim.json
   │      │  sim.pair.res
   │      │
   │      └─tmp
   │          │  code.token
   │          │  denoised.json
   │          │  sim.code.res
   │          │  sim.pair.res
   │          │  tmp.code
   │          │
   │          └─corpus
   │
   └─python
       ├─test
       │  │  cleaned.json
       │  │  final.ast
       │  │  final.code
       │  │  final.comment
       │  │  final.func_name
       │  │  final.json
       │  │  sim.code
       │  │  sim.comment
       │  │  sim.func_name
       │  │  sim.json
       │  │
       │  └─tmp
       │      │  code.token
       │      │  denoised.json
       │      │  sim.code.res
       │      │  sim.pair.res
       │      │
       │      └─corpus
       │
       ├─train
       │  │  cleaned.json
       │  │  final.ast
       │  │  final.code
       │  │  final.comment
       │  │  final.func_name
       │  │  final.json
       │  │  sim.code
       │  │  sim.comment
       │  │  sim.func_name
       │  │  sim.json
       │  │
       │  └─tmp
       │      │  code.token
       │      │  denoised.json
       │      │  sim.code.res
       │      │  sim.pair.res
       │      │
       │      └─corpus
       │
       └─valid
           │  cleaned.json
           │  final.ast
           │  final.code
           │  final.comment
           │  final.func_name
           │  final.json
           │  sim.code
           │  sim.comment
           │  sim.func_name
           │  sim.json
           │
           └─tmp
               │  code.token
               │  denoised.json
               │  sim.code.res
               │  sim.pair.res
               │
               └─corpus
```
- The tmp directory is used to store the results of Lucene processing.
- cleaned.json is the data set after the noise has been removed
- The files that start with final are the final data set, and the files that start with sim are the data related to similar codes that are matched
