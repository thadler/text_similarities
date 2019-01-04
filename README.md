# text_similarities
implementation of algorithms which compute sentence similarities

similarity_ideas.py
Use the functions in similarity_ideas.py to compute similarity matrices between 
texts. 

data_reader.py
Contains functions that read data into a format processible by 
similarity_ideas.py

preselect_ideas_for_manual_annotation.py
Implements the function of the sem_eval paper 2017 to preselect ideas "worthy" 
of being annotated by crowd workers. All other texts are assumed to have a 
similarity of zero.

sts_evaluation.py
Calculates the similarity between manual annotations and a similarity file. 
Currently only for sem_eval 2017.

experiment_....py
Files that start with experiment_... are for experiments.
