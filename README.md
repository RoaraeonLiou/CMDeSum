# CMDeSum

### 1.Process Data
1. Clean the noisy data:
   ```shell
    python filter_data.py
   ```
   The 'project_path' field in the DataFilter class needs to be changed to your project path. 
2. Create the tokenized code data file from cleaned dataset:
   ```shell
    python get_tokenized_code_file.py
   ```
3. Using Lucene to generate the similar code pairs.
4. Generate the data feature file.
   ```shell
   python generate_data_file.py
   ```
   This step will generate code files, comment files, similar code files, similar comment files, method name files, and AST files based on the search results of the previous step.
5. Generate the adjacency matrix corresponding to AST
   ```shell
   python generate_adj.py 
   ```
   Since it is slow to generate the adjacency matrix during training, we use this method to generate the adjacency matrix and save it as a file.
6. Embed the AST node
   ```shell
   python generate_node_embedding.py
   ```
   Generate node embeddings using Bert.
7. Build vocab file
   ```shell
   python build_vocab.py
   ```

### 2. Train Model
1. Step one of training
   ```shell
   python nohup python step_one_of_training.py > ./step_one.log 2>&1 &
   ```
2. Step two of training
 ```shell
   python nohup python step_two_of_training.py > ./step_one.log 2>&1 &
   ```

### 3. Test Model
```shell
python prediction.py
```
### 4. Eval Model
```shell
cd eval
python eval.py
```