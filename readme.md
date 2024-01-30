# 2024 Battle of Water Demand Forecasting
Water demand forcasting code written for WDSA/CCWI 2024 Battle of Water Networks competition. This work implements Interpretable AI's software modules, which are referenced below along with their original publications:
- Interpretable AI, LLC (2023). 'Interpretable AI Documentation.' url: https://www.interpretable.ai.
- Bertsimas, D. and Dunn, J. (2019). Machine learning under a modern optimization lens. Dynamic Ideas LLC.
- Bertsimas, D., Pawlowski, C., and Zhuo, Y. D. (2017). 'From predictive methods to missing data imputation: an optimization approach.' The Journal of Machine Learning Research, 18(1), pp:7133--7171.

Please follow Interpretable AI's installation instructions for the Julia programming language here: https://docs.interpretable.ai/stable/installation/. You will need to contact Interpretable AI to obtain a license file. If you are running VSCode, you may need to perform these additional steps:
1. Start the IAI Julia version in the REPL and run `Base.julia_cmd()`
2. Set the output of `Base.julia_cmd()` as the executable path "julia.executablePath" in your VSCode settings, e.g., 
`{
    "julia.executablePath": "/home/bradw/.julia/juliaup/julia-1.9.3+0.x64.linux.gnu/bin/julia -Cnative -J/home/bradw/.julia/artifacts/5f4ef55ac93b05a932a9f23506f5dbce9f3fe0e8/environments/Julia_v1_9_3__IAI_v3_2_0/Julia_v1_9_3__IAI_v3_2_0.so -g1"
}`

Once the Interpretable AI modules are successfully installed, run `main_script` with the following input parameters:
- `results_folder` set to the desired output folder "name_of_folder_to_save_results"
- `test_start` and `test_end` in DateTime format
- `impute_data` set to true if `IAI.ImputationLearner` is called and false otherwise
- `cp_tune` set to "auto" if we let `IAI.GridSearch` automatically calibrate the complexity parameter (recommended) and "manual" if we want to assign `cp` a fixed value (for experimental purposes)
- `lag_times` assigned as a vector of discrete lag values for model features
- `n_week_train` assigned as a vector of training windows to test
- `display_output` set to true if plotting code is activated and false otherwise 

Note that `main_script` generates DMA inflow predictions for each `n_week_train` training window size. We then apply engineering judgement to select the best model for submission.
