
########## PREAMBLE ##########

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
	include("src/functions.jl")
end

# Load time series data from CSV files
begin
	data_path = "/home/bradw/workspace/water_demand_forecasting/data/"
	inflow_df = read_data(data_path, "inflow")
	weather_df = read_data(data_path, "weather")
end

# Plot time series data
begin
	data_type = "inflow" # "inflow", "weather"
	col_names = [:dma_b] # [:dma_a]
	y_label = "Inflow [L/s]"
	start_date = DateTime("2022-01-01")
	end_date = DateTime("2022-07-23")

	if data_type == "inflow"
		plot_df = filter(row -> start_date <= row.date_time <= end_date, inflow_df)
	elseif data_type == "weather"
		plot_df = filter(row -> start_date <= row.date_time <= end_date, 	weather_df)
	end
	
	@df plot_df plot(:date_time, cols(col_names), ylabel=y_label, palette=:seaborn_bright, size=(800, 350), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9)
end



########## IMPUTE MISSING DATA ##########

"""
    Preprocessing step to fill in missing values for `inflow_df` and `weather_df` datasets.
    
    We investigate four data imputation methods:
    - mean
    - k-nearest neighbors
    - support vector machine
    - classification and regression trees
    These methods are implemented using Interpretable AI's `opt.impute` algorithm (insert citation here).

    TODO: 
    - compare data imputation methods from `impute.jl`
    - develop tailored data imputation method from Bayesian first principles
"""

# Check percentage of data with missing values
begin
	check_df = inflow_df # inflow_df, weather_df
	DataFrame(col=propertynames(check_df),
	          missing_fraction=[mean(ismissing.(col)) for col in eachcol(check_df)])
end

# Data imputation using IAI algorithms
begin
    method = :opt_knn # :zero, :mean, :opt_svm, :opt_tree, :rand
    imputer_inflow = IAI.ImputationLearner(method=method, random_seed=1)
    imputer_weather = IAI.ImputationLearner(method=method, random_seed=1)

    X_inflow = inflow_df[!, 2:end]
    X_weather = weather_df[!, 2:end]

    X_inflow = IAI.fit_transform!(imputer_inflow, X_inflow)
    X_weather = IAI.fit_transform!(imputer_weather, X_weather)

    insertcols!(X_inflow, 1, :date_time => inflow_df[!, :date_time])
    insertcols!(X_weather, 1, :date_time => weather_df[!, :date_time])
end

# Export imputed datasets to CSV file
begin
    write_data(data_path, "inflow", X_inflow)
    write_data(data_path, "weather", X_weather)
end




########## TRAIN OPTIMAL REGRESSION TREE ##########

# Create master dataframe from imputed datasets
begin
	dma_id = :dma_a
	lag_times = (1, 24)
	master_df = make_dataframe(X_inflow, X_weather, lag_times, dma_id)
    master_df = dropmissing(master_df) # missing data created by lag feature values
end

# Split train and test datasets
"""
Further exploration on the length of training data is needed.
"""

begin
    # Train dataset
    X_train = master_df[1:end-168, 3:end]
    y_train = master_df[1:end-168, 2]

    # Test dataset
    X_test = master_df[end-167:end, 3:end]
    y_test = master_df[end-167:end, 2]
end

# Run IAI optimal regression tress algorithm
"""
    Further exploration of parameter tuning is needed:
    - different performance criterion (default is :mse)
    - max tree depth
    - random_seed

"""

begin
    grid = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=1,
        ),
        max_depth=1:10,
    )
    IAI.fit!(grid, X_train, y_train, X_test, y_test)
end

# Print optimal regression tree results
begin

    # get learner
    opt_tree = IAI.get_learner(grid)

    # print summary statistics
    println(IAI.get_grid_result_summary(grid))
    println(IAI.score(grid, X_train, y_train, criterion=:mse))

    # save decision tree plot
    plot_path = "/home/bradw/workspace/water_demand_forecasting/plots/"
    IAI.write_html(plot_path * "opt_tree.html", opt_tree)
    IAI.write_svg(plot_path * "opt_tree.svg", opt_tree)

end




########## RESULTING PLOTTING AND SUMMARY STATISTICS ##########

# Compute test data performance indicators
begin
    
    # get predicted y values
    y_predict = IAI.predict(grid, X_test)

    # performance metric no. 1: mean absolute error (MAE) over first 24-h period
    mae_24h = (1/24) * sum(abs.(y_test[1:24] .- y_predict[1:24]))
    println("MAE of first 24h period: $mae_24h L/s")

    # performance metric no. 2: max absolute error over first 24-h period
    max_error_24h = maximum(abs.(y_test[1:24] .- y_predict[1:24]))
    println("Maximum absolute error of first 24h period: $max_error_24h L/s")

    # performance metric no. 1: mean absolute error (MAE) over first 24-h period
    mae_25h_to_168h = (1/144) * sum(abs.(y_test[25:168] .- y_predict[25:168]))
    println("MAE of last 144h period: $mae_25h_to_168h L/s")

end


# Plot predicted inflow data
begin

    # plot inputs
	y_label = "Inflow [L/s]"
    no_prev_steps = 3 * 168 # 3 weeks of previous inflow data

    prev_df = master_df[end-167-no_prev_steps:end, [:date_time, :dma_inflow]]
    predict_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_predict)
    actual_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_test)

	@df prev_df plot(:date_time, :dma_inflow, ylabel=y_label, color=:black, linewidth=1.25, label="Actual DMA inflow", size=(800, 350), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9, legend=:topleft)
    @df actual_df plot!(:date_time, :dma_inflow, color=:black, linewidth=1.25, label=nothing)
    @df predict_df plot!(:date_time, :dma_inflow, color=:red, linewidth=1.5, label="Predicted DMA inflow")
    vspan!([actual_df[1, :date_time], actual_df[end, :date_time]], color=:gray, alpha=0.2, label="Prediction window")
end





