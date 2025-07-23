#!/bin/bash

# RCWPS experiments
echo "Starting RCWPS experiments..."
for model in "gemini-2.5-flash-lite-preview-06-17" "gemma-3-27b-it" # "gemini-2.5-flash" 
do
  # Stack
  echo "Running Stack RCWPS with gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Stack model="$model" use_gaze=true use_ground_truth=true
  echo "Running Stack RCWPS without gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Stack model="$model" use_gaze=false use_ground_truth=false
  echo "Running Stack RCWPS with gaze and no GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Stack model="$model" use_gaze=true use_ground_truth=false
  echo "Running Stack RCWPS without gaze and with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Stack model="$model" use_gaze=false use_ground_truth=true

  # HAViD
  echo "Running HAViD RCWPS with gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=HAViD model="$model" use_gaze=true use_ground_truth=true
  echo "Running HAViD RCWPS without gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=HAViD model="$model" use_gaze=false use_ground_truth=false
  echo "Running HAViD RCWPS with gaze and no GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=HAViD model="$model" use_gaze=true use_ground_truth=false
  echo "Running HAViD RCWPS without gaze and with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=HAViD model="$model" use_gaze=false use_ground_truth=true

  # Cooking
  echo "Running Cooking RCWPS with gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Cooking model="$model" use_gaze=true use_ground_truth=true
  echo "Running Cooking RCWPS without gaze and GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Cooking model="$model" use_gaze=false use_ground_truth=false
  echo "Running Cooking RCWPS with gaze and no GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Cooking model="$model" use_gaze=true use_ground_truth=false
  echo "Running Cooking RCWPS without gaze and with GT with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=RCWPS case_study=Cooking model="$model" use_gaze=false use_ground_truth=true
done

# ICL experiments
echo "Starting ICL experiments..."
for model in "gemini-2.5-flash-lite-preview-06-17" #"gemini-2.5-flash"
do
  # Stack
  echo "Running Stack ICL with gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Stack_v2 case_study=Stack model="$model" use_gaze=true

  # HAViD
  echo "Running HAViD ICL with gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=HAViD case_study=HAViD model="$model" use_gaze=true

  # Cooking
  echo "Running Cooking ICL with gaze with model: $model"
  HYDRA_FULL_ERROR=1 python LLMcalls/run_phase_two.py exp=Cooking case_study=Cooking start_frame=4800 model="$model" use_gaze=true
done

echo "All experiments completed!"