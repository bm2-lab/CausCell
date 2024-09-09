# CausCell

## Core API interface for model training
Using this API, you can train CausCell on your own datasets using a few lines of code. 
```python
from causcell import CausCell

model = CausCell(save_and_sample_every=10)
concept_list = ['concept_A','concept_B','concept_C']
concept_counts = [4, 4, 3]
concept_cdag = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]]
results_folder = "./Output"

transformed_train_data = model.data_transformation(data_pwd="./Data/example_train.h5ad", 
                                                   save_pwd="./Data", 
                                                   concept_list=concept_list)

model.train(training_data_pwd="./Data/transformed_example_train.h5ad", 
            model_save_pwd="./Output", 
            concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, 
            training_num_steps=100)
```

## Core API interface for concept disentanglement
Using this API, you can obtain the concept representations and reconstructed cells in test dataset using a few lines of code. 
```python
from causcell import CausCell

model = CausCell()
concept_list = ['concept_A','concept_B','concept_C']
concept_counts = [4, 4, 3]
concept_cdag = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]]
results_folder = "./Output"

model.load_trained(concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, 
                   results_folder=results_folder, 
                   trained_profile_size=1000, 
                   milestone=10)

transformed_test_data = model.data_transformation(data_pwd="./Data/example_test.h5ad", 
                                                   save_pwd="./Data", 
                                                   concept_list=concept_list)

testing_data_pwd = "./Data/transformed_example_test.h5ad"


concept_embs = model.disentanglement(testing_data_pwd=testing_data_pwd, 
                                     saved_pwd="./Output", 
                                     concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag)

generated_cells = model.get_generated_cells(testing_data_pwd=testing_data_pwd, saved_pwd="./Output", 
                                            concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag)
```
## Core API interface for counterfactual generation
Using this API, you can load trained CausCell and perform counterfactual generation using a few lines of code. 
```python
from causcell import CausCell

model = CausCell()
concept_list = ['concept_A','concept_B','concept_C']
concept_counts = [4, 4, 3]
concept_cdag = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]]
results_folder = "./Output"

model.load_trained(concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, 
                   results_folder=results_folder, 
                   trained_profile_size=1000, 
                   milestone=10)

multi_target_list = [
    {"target_factor": "concept_A", "ref_factor_value":"A", "tgt_factor_value": "B"}, 
    {"target_factor": "concept_B", "ref_factor_value":"q", "tgt_factor_value": "r"}, 
]

counterfactual_generated_cells = model.counterfactual_generation(data_pwd="./Data/example_train.h5ad", 
                                                                 save_pwd='./Output', 
                                                                 concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, 
                                                                 multi_target_list=multi_target_list, 
                                                                 file_name="Counterfactual_generated_cells")
```
