##### MAIN SCRIPT FOR 2024 BWDF COMPETITION #####

include("src/functions.jl")

# Parameter selection for demand forecasting model
begin

    results_folder = "results_submission1" # "results_practice", "results_submission1"
    dma_id = :dma_j # DMA IDs a to j

    # modify datetimes accordingly
    if results_folder == "results_practice"
        test_start = DateTime("2022-07-18T00:00:00")
        test_end = DateTime("2022-07-24T23:00:00")

    elseif results_folder == "results_submission1"
        test_start = DateTime("2022-07-25T00:00:00")
        test_end = DateTime("2022-07-31T23:00:00")

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
    start_date = DateTime("2021-01-11")
    end_date = DateTime("2022-07-24")

    # data
    data_type = "inflow" # "weather"
    data_name = :dma_j # dma IDs or weather feature :air_temp

end

plot_time_series(results_folder, data_type, data_name, start_date, end_date)




