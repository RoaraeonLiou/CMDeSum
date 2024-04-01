This directory is used to store the embedded node vector and the files that map words and ids to each other. The directory structure is shown below:
```shell
csn
├─java
│  │  code.id2word
│  │  code.word2id
│  │  comment.id2word
│  │  comment.word2id
│  │
│  ├─test
│  │      adj1.pt
│  │      adj2.pt
│  │      adj3.pt
│  │      node_embedding.pt
│  │
│  ├─train
│  │      adj1.pt
│  │      adj2.pt
│  │      adj3.pt
│  │      node_embedding.pt
│  │
│  └─valid
│          adj1.pt
│          adj2.pt
│          adj3.pt
│          node_embedding.pt
│
└─python
    │  code.id2word
    │  code.word2id
    │  comment.id2word
    │  comment.word2id
    │
    ├─test
    │      adj1.pt
    │      adj2.pt
    │      adj3.pt
    │      node_embedding.pt
    │
    ├─train
    │      adj1.pt
    │      adj2.pt
    │      adj3.pt
    │      node_embedding.pt
    │
    └─valid
           adj1.pt
           adj2.pt
           adj3.pt
           node_embedding.pt
```