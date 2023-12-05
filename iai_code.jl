
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
	col_names = [:dma_c] # [:dma_a]
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
	dma_id = :dma_d
	lag_times = (1, 24, 168)
	master_df = make_dataframe(X_inflow, X_weather, lag_times, dma_id)
    master_df = dropmissing(master_df) # missing data created by lag feature values
end

# Split train and test datasets
"""
Here, we split train/test datasets for three different models: 1h, 24h, and 168h. Further exploration on the length of training data is needed.
"""

begin

    n_week_train = 10
    n_train = (1+n_week_train) * 168

    # 1h model datasets
    X_1h_train = master_df[end-n_train:end-168, 3:end]
    y_1h_train = master_df[end-n_train:end-168, 2]
    X_1h_test = master_df[end-167:end-167, 3:end]
    y_1h_test = master_df[end-167:end-167, 2]

    # 24h model datasets
    X_24h_train = master_df[end-n_train:end-168, [3:end-3; end-1:end]]
    y_24h_train = master_df[end-n_train:end-168, 2]
    X_24h_test = master_df[end-167:end-144, [3:end-3; end-1:end]]
    y_24h_test = master_df[end-167:end-144, 2]

    # 24h model datasets
    X_168h_train = master_df[end-n_train:end-168, [3:end-3; end]]
    y_168h_train = master_df[end-n_train:end-168, 2]
    X_168h_test = master_df[end-167:end, [3:end-3; end]]
    y_168h_test = master_df[end-167:end, 2]

end

# Run IAI optimal regression tress algorithm
"""
    Further exploration of parameter tuning is needed:
    - different performance criterion (default is :mse)
    - max tree depth
    - random_seed

"""

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

    # get tree learners
    opt_tree_1h = IAI.get_learner(grid_1h)
    opt_tree_24h = IAI.get_learner(grid_24h)
    opt_tree_168h = IAI.get_learner(grid_168h)

    # print summary statistics
    println(IAI.get_grid_result_summary(grid_1h))
    println(IAI.get_grid_result_summary(grid_24h))
    println(IAI.get_grid_result_summary(grid_168h))

    # save decision tree plot
    plot_path = "/home/bradw/workspace/water_demand_forecasting/plots/"
    IAI.write_html(plot_path * "opt_tree_24h.html", opt_tree_24h)
    IAI.write_svg(plot_path * "opt_tree_24h.svg", opt_tree_24h)
    IAI.write_html(plot_path * "opt_tree_168h.html", opt_tree_168h)
    IAI.write_svg(plot_path * "opt_tree_168h.svg", opt_tree_168h)

end




########## RESULTING PLOTTING AND SUMMARY STATISTICS ##########

# Compute test data performance indicators
begin
    
    # get predicted y values
    y_1h_predict = IAI.predict(grid_1h, X_1h_test)
    y_24h_predict = IAI.predict(grid_24h, X_24h_test)
    y_168h_predict = IAI.predict(grid_168h, X_168h_test)

    # metric no. 1: mean absolute error (MAE) over first day of predicted week
    mae_first_1h_24h = (1/24) * sum(abs.(y_24h_test .- vcat(y_1h_predict, y_24h_predict[2:end])))
    println("MAE of first 24h period using 1h + 24h models: $mae_first_1h_24h L/s")
    mae_first_24h = (1/24) * sum(abs.(y_24h_test .- y_24h_predict))
    println("MAE of first 24h period using 24h model: $mae_first_24h L/s")
    mae_first_168h = (1/24) * sum(abs.(y_168h_test[1:24] .- y_168h_predict[1:24]))
    println("MAE of first 24h period using 168h model: $mae_first_168h L/s")

    # metric no. 2: max absolute error over first day of predicted week
    maxAE_1h_24h = maximum(abs.(y_24h_test .- vcat(y_1h_predict, y_24h_predict[2:end])))
    println("MaxAE of first 24h period using 1h + 24h models: $maxAE_1h_24h L/s")
    maxAE_24h = maximum(abs.(y_24h_test .- y_24h_predict))
    println("MaxAE of first 24h period using 24h model: $maxAE_24h L/s")
    maxAE_168h = maximum(abs.(y_168h_test[1:24] .- y_168h_predict[1:24]))
    println("MaxAE of first 24h period using 168h model: $maxAE_168h L/s")

    # metric no. 3: mean absolute error (MAE) over first 24-h period
    mae_last_168h = (1/144) * sum(abs.(y_168h_test[25:168] .- y_168h_predict[25:168]))
    println("MAE of last 144h period using 168h model: $mae_last_168h L/s")

end


# Plot performance metrics using different models
begin

    # plotting data
    metrics = repeat(["MAE (first 24h)", "MaxAE (first 24h)", "MAE (last 144h)"], inner=3)
    vals = [float.(mae_first_1h_24h), float.(mae_first_24h), float.(mae_first_168h), float.(maxAE_1h_24h), float.(maxAE_24h), float.(maxAE_168h), NaN, NaN, float.(mae_last_168h)]
    model = repeat(["1h + 24h models", "24h model", "168h model"], outer=3)

    # plotting code
    groupedbar(metrics, vals, group=model, bar_width=0.7, xlabel="Performance metrics", ylabel="Error [L/s]", legend=:topleft)

end


# Plot predicted inflow data
begin

    # plot inputs
    colors = theme_palette(:auto).colors.colors
	y_label = "Inflow [L/s]"
    no_prev_steps = 2 * 168 # 3 weeks of previous inflow data

    prev_df = master_df[end-167-no_prev_steps:end, [:date_time, :dma_inflow]]
    actual_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_168h_test)
    predict_1h_df = DataFrame(:date_time => master_df[end-167:end-167, :date_time], :dma_inflow => y_1h_predict)
    predict_24h_df = DataFrame(:date_time => master_df[end-167:end-144, :date_time], :dma_inflow => y_24h_predict)
    predict_168h_df = DataFrame(:date_time => master_df[end-167:end, :date_time], :dma_inflow => y_168h_predict)

	@df prev_df plot(:date_time, :dma_inflow, ylabel=y_label, color=:black, linewidth=1.25, label="Actual", size=(800, 350), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9, legend=:topleft)
    @df actual_df plot!(:date_time, :dma_inflow, color=:black, linewidth=1.25, label=nothing)
    @df predict_1h_df plot!(:date_time, :dma_inflow, color=colors[2], linewidth=1.5, label="Predicted (1h model)")
    @df predict_24h_df plot!(:date_time, :dma_inflow, color=colors[3], linewidth=1.5, label="Predicted (24h model)")
    @df predict_168h_df plot!(:date_time, :dma_inflow, color=colors[1], linewidth=1.5, label="Predicted (168h model)")
    vspan!([actual_df[1, :date_time], actual_df[end, :date_time]], color=:gray, alpha=0.2, label="Prediction window")

end





