##### MAIN SCRIPT FOR 2024 BWDF COMPETITION #####

include("src/functions.jl")

# Parameter selection for demand forecasting model
begin

    results_folder = "results_practice3" # "results_practice", "results_submission1", "results_submission2"
    dma_id = :dma_e # DMA IDs a to j

    # modify datetimes accordingly
    if results_folder == "results_practice1"
        test_start = DateTime("2022-07-18T00:00:00")
        test_end = DateTime("2022-07-24T23:00:00")

    elseif results_folder == "results_submission1"
        test_start = DateTime("2022-07-25T00:00:00")
        test_end = DateTime("2022-07-31T23:00:00")

    elseif results_folder == "results_practice2"
        test_start = DateTime("2022-10-24T00:00:00")
        test_end = DateTime("2022-10-30T23:00:00")

    elseif results_folder == "results_submission2"
        test_start = DateTime("2022-10-31T00:00:00")
        test_end = DateTime("2022-11-06T23:00:00")

    elseif results_folder == "results_practice3"
        test_start = DateTime("2023-01-09T00:00:00")
        test_end = DateTime("2023-01-15T23:00:00")

    elseif results_folder == "results_submission3"
        test_start = DateTime("2023-01-16T00:00:00")
        test_end = DateTime("2023-01-22T23:00:00")

    elseif results_folder == "results_practice4"
        test_start = DateTime("2023-02-27T00:00:00")
        test_end = DateTime("2023-03-05T23:00:00")

    elseif results_folder == "results_submission4"
        test_start = DateTime("2023-03-06T00:00:00")
        test_end = DateTime("2023-03-12T23:00:00")

    end

    # Run data imputation?
    impute_data = false # run once

    # Complexity Parameter
    cp_tune = "auto" # "manual", "auto"
    cp_val = 0.001

    # Lag values for feature selection
    lag_times = [168, 24, 1] # discrete lag values

    # Training windows (no. of weeks)
    n_week_train = [52, 26, 4, 1]

    # Display results plotting?
    display_output = true # default argument is true

end

# Run main_script function
main_script(dma_id, results_folder, test_start, test_end; impute_data=impute_data, cp_tune=cp_tune, cp_val=cp_val, n_week_train=n_week_train, display_output=display_output, lag_times=lag_times)





########## Time series plotting #########

# Parameter selection
begin

    # Start and end dates
    start_date = DateTime("2022-12-27T00:00:00")
    end_date = DateTime("2022-12-29T23:00:00")

    # data
    data_type = "inflow" # "weather"
    data_name = :dma_e # dma IDs or weather feature :air_temp

end

plot_time_series(results_folder, data_type, data_name, start_date, end_date)




