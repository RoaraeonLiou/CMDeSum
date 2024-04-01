# CMDeSum
This is a replication package of CMDeSum.
### 1. Get Starded
1.1. Requirements
- OS: Any, our experiment is on Ubuntu
- GPU: At least 20G of video memory, our experiment was run on a Nvidia RTX4090 graphics card.
- Packages:
   - python 3.8
   - pytorch 1.13.0
   - torch-geometric 2.4.0
   - cuda 11.7
   - java 1.8.X
   
1.2 Dateset

Our model is evaluated on the CodeSearchNet dataset. If you want to replicate our method or use our data processing method starting with the data processing, please refer to Section 2.

In addition, if you do not want to process the data but run our code directly, we provide the processed data set at this link:


### 2.Process Data
2.1. Clean the noisy data:
   ```shell
    python filter_data.py
   ```
   The 'project_path' field in the DataFilter class needs to be changed to your project path. 
2.2. Create the tokenized code data file from cleaned dataset:
   ```shell
    python get_tokenized_code_file.py
   ```
2.3. Using Lucene to generate the similar code pairs.
2.4. Generate the data feature file.
   ```shell
   python generate_data_file.py
   ```
   This step will generate code files, comment files, similar code files, similar comment files, method name files, and AST files based on the search results of the previous step.
2.5. Generate the adjacency matrix corresponding to AST
   ```shell
   python generate_adj.py 
   ```
   Since it is slow to generate the adjacency matrix during training, we use this method to generate the adjacency matrix and save it as a file.
2.6. Embed the AST node
   ```shell
   python generate_node_embedding.py
   ```
   Generate node embeddings using Bert.
2.7. Build vocab file
   ```shell
   python build_vocab.py
   ```

### 3. Train Model
3.1. Step one of training
   ```shell
   python nohup python step_one_of_training.py > ./step_one.log 2>&1 &
   ```
3.2. Step two of training
 ```shell
   python nohup python step_two_of_training.py > ./step_one.log 2>&1 &
   ```

### 4. Test Model
```shell
python prediction.py
```
### 5. Eval Model
```shell
cd eval
python eval.py
```