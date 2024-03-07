# Load dependencies and functions located in project environment
using CSV
using DataFrames
using Dates
using Plots
using StatsPlots
using Statistics
using ColorSchemes
using Graphviz_jll
using JLD2
using Logging


function main_script(dma_id, results_folder, test_start, test_end; impute_data=true, cp_tune="auto", cp_val=nothing, lag_times=[1], n_week_train=[1], display_output=true)

    ### Step 1: load data and call data imputation method to fill in missing values
    
    data_path = pwd() * "/data/"
    results_path = pwd() * "/" * results_folder * "/"

    # Load time series data from CSV files
	inflow_df = read_data(data_path, "inflow", results_folder)
	weather_df = read_data(data_path, "weather", results_folder)

    # Data imputation
    X_inflow, X_weather = data_imputer(dma_id, results_path, impute_data, inflow_df, weather_df)


    ### Step 2: train optimal regression tree models

    # Create master dataframe from imputed datasets
	master_df = make_dataframe(X_inflow, X_weather, lag_times, dma_id)
    max_lag = maximum(lag_times)
    master_df = master_df[max_lag+1:end, :] # Drop first n rows, where n corresponds to the maximum lag value
            
    # Find indices of test period start and end
    test_start_idx = findfirst(row -> row.date_time == test_start, eachrow(master_df))
    test_end_idx = findfirst(row -> row.date_time == test_end, eachrow(master_df))


    # Loop over selected training windows and save results
    results_df = DataFrame(n_week_train = Float64[], mae_first = Float64[], maxAE = Float64[], mae_last = Float64[], total = Float64[])
    for n ∈ n_week_train

        n_train = n * 168
    
        # Create train and test datasets
        X_1h_train, y_1h_train, X_1h_test, y_1h_test = train_test_data(master_df, n_train, test_start_idx, test_end_idx, "1h")
        X_24h_train, y_24h_train, X_24h_test, y_24h_test = train_test_data(master_df, n_train, test_start_idx, test_end_idx, "24h")
        X_168h_train, y_168h_train, X_168h_test, y_168h_test = train_test_data(master_df, n_train, test_start_idx, test_end_idx, "168h")


        # Run IAI optimal regression tress algorithm
        cpu_time = @elapsed begin

            if cp_tune == "auto"
                
                # 1h model
                grid_1h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                )
                IAI.fit!(grid_1h, X_1h_train, y_1h_train)

                # 24h model
                grid_24h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                )
                IAI.fit!(grid_24h, X_24h_train, y_24h_train)

                # 168h model
                grid_168h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                )
                IAI.fit!(grid_168h, X_168h_train, y_168h_train)

            elseif cp_tune == "manual"

                # 1h model
                grid_1h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                    cp=cp_val,
                )
                IAI.fit!(grid_1h, X_1h_train, y_1h_train)

                # 24h model
                grid_24h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                    cp=cp_val,
                )
                IAI.fit!(grid_24h, X_24h_train, y_24h_train)

                # 168h model
                grid_168h = IAI.GridSearch(
                    IAI.OptimalTreeRegressor(
                        random_seed=2,
                    ),
                    max_depth=1:8,
                    cp=cp_val,
                )
                IAI.fit!(grid_168h, X_168h_train, y_168h_train)

            end


        end

        # Get tree learners
        opt_tree_1h = IAI.get_learner(grid_1h)
        opt_tree_24h = IAI.get_learner(grid_24h)
        opt_tree_168h = IAI.get_learner(grid_168h)

        # Save decision tree plots
        plot_path = pwd() * "/" * results_folder * "/plots/"
        IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_1h_train_" * string(n) * ".svg", opt_tree_1h)
        IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_24h_train_" * string(n) * ".svg", opt_tree_24h)
        IAI.write_svg(plot_path * string(dma_id) * "_opt_tree_168h_train_" * string(n) * ".svg", opt_tree_168h)

        # Compute performance metrics and save best forecasted demand time series
        save_results(results_path, results_df, grid_1h, grid_24h, grid_168h, X_1h_test, X_24h_test, X_168h_test, y_168h_test, n, dma_id)

    end


    # ### Step 3: results plotting

    # Plot forecasted demands
    plot_forecast(results_path, master_df, test_start_idx, test_end_idx, n_week_train, display_output, dma_id)

    # Save and print results_df
    CSV.write(results_path * "demand_forecast/" * string(dma_id) * "_results_metrics.csv", results_df)
    if display_output
        display(results_df)
    end


end
    


###################################################################################################################


function read_data(data_path::String, data_type::String, results_folder::String)

if results_folder == "results_submission1" || results_folder == "results_practice1"

    df_mean = DataFrame()

    if data_type == "inflow"
        df = CSV.read(data_path * "InflowData_1.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    elseif data_type == "weather"
        df = CSV.read(data_path * "WeatherData_1.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")
    end

    # add data at spring 2022 time change
    n = 10802
    df_insert = df[n-1:n+1, 2:end]
    try
        df_mean = mean.(skipmissing.(eachcol(df_insert)))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
    end
    insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

    # delete duplicate data from autumn 2021 time change
    n = 7274
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2021 time change
    n = 2066
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

elseif results_folder == "results_submission2" || results_folder == "results_practice2"

    df_mean = DataFrame()

    if data_type == "inflow"
        df = CSV.read(data_path * "InflowData_2.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    elseif data_type == "weather"
        df = CSV.read(data_path * "WeatherData_2.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")
    end

    # delete duplicate data from autumn 2022 time change
    n = 16010
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2022 time change
    n = 10802
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

    # delete duplicate data from autumn 2021 time change
    n = 7274
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2021 time change
    n = 2066
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

elseif results_folder == "results_submission3" || results_folder == "results_practice3"

    df_mean = DataFrame()

    if data_type == "inflow"
        df = CSV.read(data_path * "InflowData_3.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    elseif data_type == "weather"
        df = CSV.read(data_path * "WeatherData_3.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")
    end

    # delete duplicate data from autumn 2022 time change
    n = 16010
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2022 time change
    n = 10802
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

    # delete duplicate data from autumn 2021 time change
    n = 7274
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2021 time change
    n = 2066
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])


elseif results_folder == "results_submission4" || results_folder == "results_practice4"

    df_mean = DataFrame()

    if data_type == "inflow"
        df = CSV.read(data_path * "InflowData_4.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    elseif data_type == "weather"
        df = CSV.read(data_path * "WeatherData_4.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")
    end

    # delete duplicate data from autumn 2022 time change
    n = 16010
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2022 time change
    n = 10802
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

    # delete duplicate data from autumn 2021 time change
    n = 7274
    df_delete = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_delete)))
    try 
        df[n, 2:end] = df_mean
    catch
        df_mean = median.(skipmissing.(eachcol(df_delete)))
        df[n, 2:end] = df_mean
    end
    delete!(df, n+1)

    # add data at spring 2021 time change
    n = 2066
    df_insert = df[n-1:n+1, 2:end]
    df_mean = mean.(skipmissing.(eachcol(df_insert)))
    try
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    catch
        df_mean = median.(skipmissing.(eachcol(df_insert)))
        insert_data = DataFrame(hcat(DateTime(2022, 3, 27, 2, 0, 0), df_mean'), names(df))
    end
    df = vcat(df[1:n, :], insert_data, df[n+1:end, :])

end


    return df
end


function write_data(data_path::String, data_type::String, df)

    if data_type == "inflow"
        CSV.write(data_path * "inflow_imputed.csv", df)

    elseif data_type == "weather"
        CSV.write(data_path * "weather_imputed.csv", df)

    end

end


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
function data_imputer(dma_id, results_path, impute_data, inflow_df, weather_df)

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

        X_inflow[!, 2:end] .= round.(X_inflow[!, 2:end], digits=2)
        X_weather[!, 2:end] = round.(X_weather[!, 2:end], digits=2)

        # Export imputed datasets to CSV file
        CSV.write(results_path * "imputed_data/inflow_imputed.csv", X_inflow)
        CSV.write(results_path * "imputed_data/weather_imputed.csv", X_weather)

    else

        try
            # Load imputed data from results folder
            X_inflow = CSV.read(results_path * "imputed_data/inflow_imputed.csv", DataFrame)
            X_weather = CSV.read(results_path * "imputed_data/weather_imputed.csv", DataFrame)
        catch
            @error "No data imputation csv files exist."
        end

    end


    return X_inflow, X_weather
end


function make_dataframe(inflow_df, weather_df, lag_times, dma_id)

    # define holiday dates
    holiday_dates = [Dates.Date("2021-01-01"), Dates.Date("2021-01-06"), Dates.Date("2021-04-04"), Dates.Date("2021-04-05"), Dates.Date("2021-04-25"), Dates.Date("2021-05-01"), Dates.Date("2021-06-02"), Dates.Date("2021-08-15"), Dates.Date("2021-11-01"), Dates.Date("2021-11-03"), Dates.Date("2021-12-08"), Dates.Date("2021-12-25"), Dates.Date("2021-12-26"), Dates.Date("2022-01-01"), Dates.Date("2022-01-06"), Dates.Date("2022-04-17"), Dates.Date("2021-04-18"), Dates.Date("2022-04-25"), Dates.Date("2022-05-01"), Dates.Date("2022-06-02"), Dates.Date("2022-08-15"), Dates.Date("2022-11-01"), Dates.Date("2022-11-03"), Dates.Date("2022-12-08"), Dates.Date("2022-12-25"), Dates.Date("2022-12-26"), Dates.Date("2023-01-01"), Dates.Date("2023-01-06")]

    # make new dataframe and add time features
    df_time = DataFrame()

    df_time.date_time = weather_df.date_time
    df_time.quarter = ceil.(Int, Dates.month.(weather_df.date_time) / 3)    
    df_time.month = Dates.month.(weather_df.date_time)
    # df_time.week_of_month = Dates.dayofweekofmonth.(weather_df.date_time)
    df_time.day_of_week = Dates.dayofweek.(weather_df.date_time)
    day_of_week = Dates.dayofweek.(weather_df.date_time)
    df_time.day_type = [(day ∈ 1:5) && !(date in holiday_dates) ? 1 : 0 for (day, date) in zip(day_of_week, df_time.date_time)]
    df_time.time = Dates.value.(Dates.Hour.(weather_df.date_time))
    # df_time.is_holiday = [date in holiday_dates ? 1 : 0 for date in df_time.date_time]

    # merge weather data
    df_feat = DataFrame()
    df_feat = unique(rightjoin(df_time, weather_df, on=:date_time))

    # select DMA to analyze and merge feature and inflow dataframes
    df = DataFrame()
    df = outerjoin(inflow_df[!, [:date_time, dma_id]], df_feat, on=:date_time)

    # rename dma inflow column
    rename!(df, Dict(dma_id => :dma_inflow))

    # # minimum night flow
    # mnf = 24
    # df[!, Symbol("min_", mnf, "h_inflow")] = [fill(missing, mnf); [minimum(df.dma_inflow[j-mnf:j]) for j ∈ mnf+1:length(df.dma_inflow)]]

    # lagged values
    for (i, v) ∈ enumerate(lag_times)
            df[!, Symbol("prev_", v, "h_inflow")] = [fill(missing, v); df.dma_inflow[1:end - v]]
    end

    return df
    
end


"""
Model Training:
    We split train/test datasets for three different models: 24h, and 168h, and use the previous n weeks as the training window.
    
    TO-DO: 
        - Further exploration on the duration (and time of year) of the training window.

"""
function train_test_data(master_df, n_train, test_start_idx, test_end_idx, model::String)

    # # set mnf values for test dataset = last time step in train dataset
    # last_mnf = master_df[test_start_idx-1, "min_24h_inflow"] 
    # master_df[test_start_idx:test_end_idx, "min_24h_inflow"] .= last_mnf

    if model == "1h"

        X_train = master_df[test_start_idx-n_train:test_start_idx-1, 3:end]
        y_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
        X_test = master_df[test_start_idx:test_start_idx, 3:end]
        y_test = master_df[test_start_idx:test_start_idx, 2]

    elseif model == "24h"

        X_train = master_df[test_start_idx-n_train:test_start_idx-1, 3:end-1]
        y_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
        X_test = master_df[test_start_idx:test_start_idx+23, 3:end-1]
        y_test = master_df[test_start_idx:test_start_idx+23, 2]

    elseif model == "168h"

        X_train = master_df[test_start_idx-n_train:test_start_idx-1, 3:end-2]
        y_train = master_df[test_start_idx-n_train:test_start_idx-1, 2]
        X_test = master_df[test_start_idx:test_end_idx, 3:end-2]
        y_test = master_df[test_start_idx:test_end_idx, 2]

    end

    return X_train, y_train, X_test, y_test

end


function save_results(results_path, results_df, grid_1h, grid_24h, grid_168h, X_1h_test, X_24h_test, X_168h_test, y_168h_test, n, dma_id)

    # Get predicted y values
    y_1h_predict = []
    y_24h_predict = []
    y_168h_predict = []

    y_1h_predict = IAI.predict(grid_1h, X_1h_test)
    y_24h_predict = IAI.predict(grid_24h, X_24h_test)
    y_168h_predict = IAI.predict(grid_168h, X_168h_test)

    # Create all predicted y value combinations
    y_predict = vcat(y_1h_predict, y_24h_predict[2:end], y_168h_predict[25:end])

    # Compute performance metrics
    mae_first = []
    maxAE = []
    mae_last = []

    try
        mae_first = (1/24) * sum(abs.(y_168h_test[1:24] .- y_predict[1:24]))
        maxAE = maximum(abs.(y_168h_test[1:24] .- y_predict[1:24]))
        mae_last = (1/144) * sum(abs.(y_168h_test[25:168] .- y_predict[25:168]))

        # Save performance metrics to results dataframe
        push!(results_df, [n, mae_first, maxAE, mae_last, mae_first+maxAE+mae_last])
    catch
        println("Test data not provided. Cannot compute performance metrics.")
        mae_first = []
        maxAE = []
        mae_last = []
    end

    # Save y_predict data to csv
    CSV.write(results_path * "demand_forecast/" * string(dma_id) * "_inflow_predict_" * string(n) * ".csv", DataFrame(y_predict=y_predict))

end



function plot_forecast(results_path, master_df, test_start_idx, test_end_idx, n_week_train, display_output, dma_id)

    predict_df = DataFrame(:date_time => master_df[test_start_idx:test_end_idx, :date_time])
    actual_df = DataFrame(:date_time => master_df[test_start_idx:test_end_idx, :date_time], :actual_inflow => master_df[test_start_idx:test_end_idx, 2])
    col_names = []

    # Load y_predict data
    for n ∈ n_week_train

        column_name = Symbol("predict_$n")
        append!(col_names, [column_name])
        data = CSV.read(results_path * "demand_forecast/" * string(dma_id) * "_inflow_predict_" * string(n) * ".csv", DataFrame)
        predict_df[!, column_name] = data[!, "y_predict"]

    end

    # Plotting code
    plt = @df predict_df plot(:date_time, cols(col_names), palette=:seaborn_bright, linewidth=1.25)
    plt = @df actual_df plot!(:date_time, :actual_inflow, color=:black, linewidth=1.5, label="actual", size=(1000, 400), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9, legend=:outertopright, ylabel="Inflow [L/s]")

    if display_output
        display(plt)
    end

    savefig(results_path * "plots/" * string(dma_id) * "_inflow_predict.svg")

end


function plot_time_series(results_folder, data_type, data_name, start_date, end_date)

    results_path = pwd() * "/" * results_folder * "/" 

    inflow_df = CSV.read(results_path * "imputed_data/inflow_imputed.csv", DataFrame)
    weather_df = CSV.read(results_path * "imputed_data/weather_imputed.csv", DataFrame)

    if data_type == "inflow"
        plot_df = filter(row -> start_date <= row.date_time <= end_date, inflow_df)
        y_label = "Inflow [L/s]"
        @df plot_df plot(:date_time, cols([data_name]), ylabel=y_label, palette=:seaborn_bright, size=(1000, 400), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9)

    elseif data_type == "weather"
        plot_df = filter(row -> start_date <= row.date_time <= end_date, weather_df)
        y_label = data_name
        @df plot_df plot(:date_time, cols([data_name]), ylabel=y_label, palette=:seaborn_bright, size=(1000, 400), xguidefontsize=10, xtickfontsize=9, yguidefontsize=10, ytickfontsize=9, legendfontsize=9)

    end


end
