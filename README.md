# ResearchSmear-
Working with HMM models to analyze data in large batches 


README Written by Trevor Robbins
Below is a list of benchmarks in the analysis of Nose_x, Nose_y parameters for the model.
1) Transition matrix has been defined and produced from the model's inputs. 
2) Zero sparsing. Had to minimize and optimize model to ensure range of data was minimal. Range was squeezed down to 2.1%
3) CUDA Compiler failing. PTXAS failed during a 11-hour run, had to restart and discard data, fixed in further iterations.

