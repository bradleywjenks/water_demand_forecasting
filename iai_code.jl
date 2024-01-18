# Load dependencies and functions located in project environment
begin
	using CSV
	using DataFrames
	using Dates
	using Plots
	using StatsPlots
	using Statistics
	using ColorSchemes
    using Graphviz_jll
    using JLD2
	include("src/functions.jl")
end



########## PREAMBLE ##########

"""
    First step in this work:
    - select DMA to forecast demand
    - load csv data
    - plot data (optional)
 
"""

# Select results folder, DMA, and forecasting period
begin

    results_folder = "results_practice" # "results_practice", "results_submission1"
    dma_id = :dma_a # DMA IDs a to j

    if results_folder == "results_practice"
        test_start = DateTime("2022-07-18T00:00:00")
        test_end = DateTime("2022-07-24T23:00:00")

    # modify datetimes accordingly
    elseif results_folder == "results_submission1"
        test_start = DateTime("2022-07-24T00:00:00")
        test_end = DateTime("2022-07-31T23:00:00")

    end
end

# Load time series data from CSV files
begin
	data_path = pwd() * "/data/"
	inflow_df = read_data(data_path, "inflow")
	weather_df = read_data(data_path, "weather")
end

# Plot time series data
begin
	data_type = "inflow" # "inflow", "weather"
	col_names = [dma_id]
	y_label = "Inflow [L/s]"
	start_date = DateTime("2021-01-01")
	end_date = DateTime("2022-07-23")

	if data_type == "inflow"
		plot_df = filter(row -> start_date <= row.date_time <= end_date, inflow_df)
	elseif data_type == "weather"
		plot_df = filter(row -> start_date <= row.date_time <= end_date, 	weather_df)
	end
	
	@df plot_df plot(:date_time, cols(col_names), ylabel=y_label, palette=:seaborn_bright, size=(500, 250), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9)
end



########## IMPUTE MISSING DATA ##########

"""
    Preprocessing step to fill in missing values for `inflow_df` and `weather_df` datasets.
    
    We investigate four data imputation methods:
    - mean
    - k-nearest neighbors
    - support vector machine
    - classification and regression trees
    These methods are implemented using Interpretable AI's `opt.impute` algorithm.

    TO-DO: 
    - compare data imputation methods from `impute.jl`
    - develop tailored data imputation method from Bayesian first principles

    NOTE: THIS STEP SHOULD NOT BE COMPLETED FOR REPRODUCING RESULTS. INSTEAD, PLEASE SET `IMPUTE_DATA` TO FALSE 
"""

# Data imputation
begin

    # Run data imputation?
    impute_data = true
    results_path = pwd() * "/" * results_folder * "/"

    if impute_data

        # Check percentage of data with missing values
        check_df = inflow_df # inflow_df, weather_df
        DataFrame(col=propertynames(check_df),
        missing_fraction=[mean(ismissing.(col)) for col in eachcol(check_df)])

        # Data imputation using IAI algorithms
        method = :opt_knn # :zero, :mean, :opt_svm, :opt_tree, :rand
        imputer_inflow = IAI.ImputationLearner(method=method, random_seed=1)
        imputer_weather = IAI.ImputationLearner(method=method, random_seed=1)

        X_inflow = inflow_df[!, 2:end]
        X_weather = weather_df[!, 2:end]

        X_inflow = IAI.fit_transform!(imputer_inflow, X_inflow)
        X_weather = IAI.fit_transform!(imputer_weather, X_weather)

        insertcols!(X_inflow, 1, :date_time => inflow_df[!, :date_time])
        insertcols!(X_weather, 1, :date_time => weather_df[!, :date_time])

        # Export imputed datasets to CSV file
        CSV.write(results_path * "imputed_data/" * string(dma_id) * "_inflow_imputed.csv", X_inflow)
        CSV.write(results_path * "imputed_data/" * string(dma_id) * "_weather_imputed.csv", X_weather)

    else

        # Load imputed data from results folder
        X_inflow = CSV.read(results_path * "imputed_data/" * string(dma_id) * "_inflow_imputed.csv", DataFrame)
        X_weather = CSV.read(results_path * "imputed_data/" * string(dma_id) * "_weather_imputed.csv", DataFrame)

    end

end




########## TRAIN OPTIMAL REGRESSION TREE ##########


# Create master dataframe from imputed datasets
begin

	lag_times = (1, 24, 168)
	master_df = make_dataframe(X_inflow, X_weather, lag_times, dma_id)

    # Drop first n rows, where n corresponds to the maximum lag value
    max_lag = maximum(lag_times)
    master_df = master_df[max_lag+1:end, :]

end

"""

    Here, we split train/test datasets for three different models: 1h, 24h, and 168h. We simply use the previous n weeks as the training window.
    
    TO-DO: 
        - Further exploration on the duration (and time of year) of the training window.

"""

begin 

    # No. of training weeks
    n_week_train = 52 # 1, 4, 13, 26, 52 
    n_train = (n_week_train) * 168

    # Find indices of test period start and end
    test_start_idx = findfirst(row -> row.date_time == test_start, eachrow(master_df))
    test_end_idx = findfirst(row -> row.date_time == test_end, eachrow(master_df))

end

begin

    # 1h model datasets
    X_1h_train = master_df[test_start_idx-n_train:test_start_idx-1, 3:end]
    y_1h_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
    X_1h_test = master_df[test_start_idx:test_end_idx-lag_times[3]+1, 3:end]
    y_1h_test = master_df[test_start_idx:test_end_idx-lag_times[3]+1, 2]

    # 24h model datasets
    X_24h_train = master_df[test_start_idx-n_train:test_start_idx-1, [3:end-3; end-1:end]]
    y_24h_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
    X_24h_test = master_df[test_start_idx:test_start_idx+lag_times[2]-1, [3:end-3; end-1:end]]
    y_24h_test = master_df[test_start_idx:test_start_idx+lag_times[2]-1, 2]

    # 168h model datasets
    X_168h_train = master_df[test_start_idx-n_train:test_start_idx-1, [3:end-3; end]]
    y_168h_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
    X_168h_test = master_df[test_start_idx:test_end_idx, [3:end-3; end]]
    y_168h_test = master_df[test_start_idx:test_end_idx, 2]

end

# Run IAI optimal regression tress algorithm
cpu_time = @elapsed begin

    # 1h model
    grid_1h = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=2,
        ),
        max_depth=1:10,
    )
    IAI.fit!(grid_1h, X_1h_train, y_1h_train, X_1h_test, y_1h_test)

    # 24h model
    grid_24h = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=2,
        ),
        max_depth=1:10,
    )
    IAI.fit!(grid_24h, X_24h_train, y_24h_train, X_24h_test, y_24h_test)

    # 168h model
    grid_168h = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=2,
        ),
        max_depth=1:10,
    )
    IAI.fit!(grid_168h, X_168h_train, y_168h_train, X_168h_test, y_168h_test)

end

# Print optimal regression tree results
begin

    # Get tree learners
    opt_tree_1h = IAI.get_learner(grid_1h)
    opt_tree_24h = IAI.get_learner(grid_24h)
    opt_tree_168h = IAI.get_learner(grid_168h)

    # Print summary statistics
    println(IAI.get_grid_result_summary(grid_1h))
    println(IAI.get_grid_result_summary(grid_24h))
    println(IAI.get_grid_result_summary(grid_168h))

    # Save decision tree plot
    plot_path = pwd() * "/" * results_folder * "/plots/"
    IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_1h_train_" * string(n_week_train) * ".svg", opt_tree_1h)
    IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_24h_train_" * string(n_week_train) * ".svg", opt_tree_24h)
    IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_168h_train_" * string(n_week_train) * ".svg", opt_tree_168h)

end




########## RESULTS PLOTTING AND SUMMARY STATISTICS ##########

# Compute test data performance indicators
begin
    
    # Get predicted y values
    y_1h_predict = IAI.predict(grid_1h, X_1h_test)
    y_24h_predict = IAI.predict(grid_24h, X_24h_test)
    y_168h_predict = IAI.predict(grid_168h, X_168h_test)

    # Create all predicted y value combinations
    y1_predict = vcat(y_1h_predict, y_24h_predict[2:end], y_168h_predict[25:end])
    y2_predict = vcat(y_24h_predict, y_168h_predict[25:end])
    y3_predict = y_168h_predict

    # Metric no. 1: mean absolute error (MAE) over first day of predicted week
    mae_first_y1 = (1/24) * sum(abs.(y_168h_test[1:24] .- y1_predict[1:24]))
    mae_first_y2 = (1/24) * sum(abs.(y_168h_test[1:24] .- y2_predict[1:24]))
    mae_first_y3 = (1/24) * sum(abs.(y_168h_test[1:24] .- y3_predict[1:24]))

    # Metric no. 2: max absolute error over first day of predicted week
    maxAE_y1 = maximum(abs.(y_168h_test[1:24] .- y1_predict[1:24]))
    maxAE_y2 = maximum(abs.(y_168h_test[1:24] .- y2_predict[1:24]))
    maxAE_y3 = maximum(abs.(y_168h_test[1:24] .- y3_predict[1:24]))

    # Metric no. 3: mean absolute error (MAE) over first 24-h period
    mae_last = (1/144) * sum(abs.(y_168h_test[25:168] .- y3_predict[25:168]))

    # Aggregate metrics
    y1_score = mae_first_y1 + maxAE_y1 + mae_last
    y2_score = mae_first_y2 + maxAE_y2 + mae_last
    y3_score = mae_first_y3 + maxAE_y3 + mae_last

    y_scores = vcat(y1_score, y2_score, y3_score)

    # Get predicted data and metrics for best model
    best_model = argmin(y_scores)

    if best_model == 1
        y_predict = y1_predict
        mae_first = mae_first_y1
        maxAE = maxAE_y1
        mae_last = mae_last

    elseif best_model == 2
        y_predict = y2_predict
        mae_first = mae_first_y2
        maxAE = maxAE_y2
        mae_last = mae_last

    else
        y_predict = y3_predict
        mae_first = mae_first_y3
        maxAE = maxAE_y3
        mae_last = mae_last
    end

    # Save best predicted data to csv
    CSV.write(results_path * "demand_forecast/" * string(dma_id) * "_inflow_train_" * string(n_week_train) * ".csv", DataFrame(y_predict=y_predict))

    # Save IAI models and performance metrics to julia data file
    # @save results_path * string(dma_id) * "_results_train_" * string(n_week_train) * ".jld2" grid_1h grid_24h grid_168h mae_first maxAE mae_last
    @save results_path * "demand_forecast/" * string(dma_id) * "_results_train_" * string(n_week_train) * ".jld2" mae_first maxAE mae_last

end

# Plot predicted inflow data
begin

    actual_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_168h_test)
    predict_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_predict)

    plt = @df actual_df plot(:date_time, :dma_inflow, color=:gray, linewidth=1.25, label="Actual", size=(500, 250), ylims=(0, 20), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9, legend=:topright, ylabel="Inflow [L/s]")
    plt = @df predict_df plot!(:date_time, :dma_inflow, color=:blue, linewidth=1.5, label="Predicted")
    # vspan!([actual_df[1, :date_time], actual_df[end, :date_time]], color=:gray, alpha=0.2, label="Prediction window")
   display(plt)

    savefig(results_path * "plots/" * string(dma_id) * "_inflow_train_" * string(n_week_train) * ".svg")

end





