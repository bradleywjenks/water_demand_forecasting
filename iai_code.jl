
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
	
	@df plot_df plot(:date_time, cols(col_names), ylabel=y_label, palette=:Set1_5, size=(700, 400))
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
	lag_times = (1, 24, 168)
	master_df = make_dataframe(X_inflow, X_weather, lag_times, dma_id)
    dropmissing(master_df) # missing data created by lag feature values
end

# Split train and test datasets
begin
    
end













# run IAI algorithm (train)
begin

    # define test and train data
    X = df[:, 3:end]
    y = df[:, 2]

    (train_X, train_y), (test_X, test_y) = IAI.split_data(:regression, X, y, seed=12345)

    grid = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=123,
        ),
        max_depth=1:5,
    )
    IAI.fit!(grid, train_X, train_y)
    IAI.get_learner(grid)

end

# run IAI algorithm (test)
begin

    predict_y = IAI.predict(grid, test_X)

    print(IAI.score(grid, train_X, train_y, criterion=:mse))
    print(IAI.score(grid, test_X, test_y, criterion=:mse))

    predict_y

    # # comparison plot
    time = 1:length(test_y)
    # plt = plot(time, test_y, label="Actual")
    # plt = plot!(time, predict_y, label="Predict")
    plt = plot(time[1:168], test_y[1:168], label="Actual")
    plt = plot!(time[1:168], predict_y[1:168], label="Predict")

    plt = xlabel!("Time")
    plt = ylabel!("Value")
    plt = title!("Time Series Plot")

    # Show the plot
    display(plt)

    # write results to csv file
    # results_df = DataFrame(Time=time, Series1=test_y, Series2=predict_y)
    # CSV.write(data_path * "results.csv", results_df)
end