#!/bin/bash

# RCWPS experiments
echo "Starting RCWPS experiments with ablations"
for model in "gemini-2.5-flash-lite" "gemma-3-27b-it" 
do
  # Stack
  echo "Running Stack RCWPS with gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true use_ground_truth=true exp.type=RCWPS
  echo "Running Stack RCWPS without gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=false use_ground_truth=false exp.type=RCWPS
  echo "Running Stack RCWPS with gaze and no GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true use_ground_truth=false exp.type=RCWPS
  echo "Running Stack RCWPS without gaze and with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=false use_ground_truth=true exp.type=RCWPS

  # HAViD
  echo "Running HAViD RCWPS without GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false use_ground_truth=false exp.type=RCWPS
  echo "Running HAViD RCWPS with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false use_ground_truth=true exp.type=RCWPS

  # Cooking
  echo "Running Cooking RCWPS with gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true use_ground_truth=true exp.type=RCWPS exp.use_ego=true
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true use_ground_truth=true exp.type=RCWPS exp.use_ego=false
  echo "Running Cooking RCWPS without gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=false use_ground_truth=false exp.type=RCWPS exp.use_ego=true
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=false use_ground_truth=false exp.type=RCWPS exp.use_ego=false
  echo "Running Cooking RCWPS with gaze and no GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true use_ground_truth=false exp.type=RCWPS exp.use_ego=true
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true use_ground_truth=false exp.type=RCWPS exp.use_ego=false
  echo "Running Cooking RCWPS without gaze and with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=false use_ground_truth=true exp.type=RCWPS exp.use_ego=true
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=false use_ground_truth=true exp.type=RCWPS exp.use_ego=false
done

# for model in "gemini-2.5-flash"
# do
#   # Stack
#   echo "Running Stack RCWPS with gaze and GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model="$model" use_gaze=true use_ground_truth=true exp.type=RCWPS
#   echo "Running Stack RCWPS without gaze and GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model="$model" use_gaze=false use_ground_truth=false exp.type=RCWPS
#   echo "Running Stack RCWPS with gaze and no GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model="$model" use_gaze=true use_ground_truth=false exp.type=RCWPS
#   echo "Running Stack RCWPS without gaze and with GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model="$model" use_gaze=false use_ground_truth=true exp.type=RCWPS

#   # HAViD
#   echo "Running HAViD RCWPS without GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model="$model" use_gaze=false use_ground_truth=false exp.type=RCWPS
#   echo "Running HAViD RCWPS with GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model="$model" use_gaze=false use_ground_truth=true exp.type=RCWPS

#   # Cooking
#   echo "Running Cooking RCWPS with gaze and GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=true use_ground_truth=true exp.type=RCWPS exp.use_ego=true
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=true use_ground_truth=true exp.type=RCWPS exp.use_ego=false
#   echo "Running Cooking RCWPS without gaze and GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=false use_ground_truth=false exp.type=RCWPS exp.use_ego=true
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=false use_ground_truth=false exp.type=RCWPS exp.use_ego=false
#   echo "Running Cooking RCWPS with gaze and no GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=true use_ground_truth=false exp.type=RCWPS exp.use_ego=true
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=true use_ground_truth=false exp.type=RCWPS exp.use_ego=false
#   echo "Running Cooking RCWPS without gaze and with GT with model: $model"
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=false use_ground_truth=true exp.type=RCWPS exp.use_ego=true
#   HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model="$model" use_gaze=false use_ground_truth=true exp.type=RCWPS exp.use_ego=false
# done

# ICL experiments
echo "Starting ICL experiments..."
for model in "gemini-2.5-flash-lite" 
do
  # Stack
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true exp.type=ICL
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=false exp.type=ICL

  # HAViD
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false exp.type=ICL

  # Cooking
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=ICL exp.use_ego=true
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=ICL exp.use_ego=false
done

for model in "gemini-2.5-flash"
do
  # Stack
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true exp.type=ICL

  # HAViD
  echo "Running HAViD ICL with gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false exp.type=ICL

  # Cooking
  echo "Running Cooking ICL with gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=ICL exp.use_ego=true
done

# Phase2 experiments


echo "Starting Phase2 experiments with Gemini 2.5"
for model in "gemini-2.5-flash"
do
  # Stack Phase2
  echo "Running Stack Phase2 with 2 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true exp.type=phase2 num_examples=2
  
  # HAViD Phase2
  echo "Running HAViD Phase2 with 2 examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false exp.type=phase2 num_examples=2
  
  # Cooking Phase2
  echo "Running Cooking Phase2 with 2 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=phase2 num_examples=2 exp.use_ego=true
done

for model in "gemma-3-27b-it" 
do
  # Stack Phase2
  echo "Running Stack Phase2 with 1 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true exp.type=phase2 num_examples=1
  
  # HAViD Phase2
  echo "Running HAViD Phase2 with 1 examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false exp.type=phase2 num_examples=1
  
  # Cooking Phase2
  echo "Running Cooking Phase2 with 1 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=phase2 num_examples=1 exp.use_ego=true
done

# Phase 2 ablations 
echo "Starting Phase2 experiments gemini-2.5-flash-lite-preview-06-17"
for model in "gemini-2.5-flash-lite-preview-06-17"
do
  # Stack Phase2
  echo "Running Stack Phase2 with 2 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=true exp.type=phase2 num_examples=2
  echo "Running Stack Phase2 with no examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model exp.type=phase2 num_examples=0
  echo "Running Stack Phase2 with 2 examples and no gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model=$model use_gaze=false exp.type=phase2 num_examples=2

  # HAViD Phase2
  echo "Running HAViD Phase2 with 2 examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model use_gaze=false exp.type=phase2 num_examples=2
  echo "Running HAViD Phase2 with no examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model=$model exp.type=phase2 num_examples=0 use_gaze=false 

  # Cooking Phase2
  echo "Running Cooking Phase2 with 2 examples and gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=true exp.type=phase2 num_examples=1 exp.use_ego=true
  echo "Running Cooking Phase2 with no examples with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model exp.type=phase2 num_examples=0 exp.use_ego=true
  echo "Running Cooking Phase2 with 2 examples and no gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking model=$model use_gaze=false exp.type=phase2 num_examples=1 exp.use_ego=true
done

echo "All experiments completed!"