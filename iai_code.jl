
using CSV
using IJulia
using DataFrames
using Dates
using Plots

begin

    # load data
    data_path = "/home/bradw/workspace/water_demand_forecasting/data/"

    inflow_df = CSV.read(data_path * "InflowData_1.csv", DataFrame)
    inflow_df.date_time = Dates.DateTime.(inflow_df.date_time, "dd/mm/yyyy HH:MM")

    weather_df = CSV.read(data_path * "WeatherData_1.csv", DataFrame)
    weather_df.date_time = Dates.DateTime.(weather_df.date_time, "dd/mm/yyyy HH:MM")

    # define holiday dates
    holiday_dates = [Dates.Date("2021-01-01"), Dates.Date("2021-01-06"), Dates.Date("2021-04-04"), Dates.Date("2021-04-05"), Dates.Date("2021-04-25"), Dates.Date("2021-05-01"), Dates.Date("2021-06-02"), Dates.Date("2021-08-15"), Dates.Date("2021-11-01"), Dates.Date("2021-11-03"), Dates.Date("2021-12-08"), Dates.Date("2021-12-25"), Dates.Date("2021-12-26"), Dates.Date("2022-01-01"), Dates.Date("2022-01-06"), Dates.Date("2022-04-17"), Dates.Date("2021-04-18"), Dates.Date("2022-04-25"), Dates.Date("2022-05-01"), Dates.Date("2022-06-02"), Dates.Date("2022-08-15")]

end

begin

    # make new dataframe and add time features
    df_time = DataFrame()

    df_time.date_time = weather_df.date_time
    df_time.quarter = ceil.(Int, Dates.month.(weather_df.date_time) / 3)
    df_time.month = Dates.month.(weather_df.date_time)
    df_time.week_of_month = Dates.dayofweekofmonth.(weather_df.date_time)
    df_time.day_of_week = Dates.dayofweek.(weather_df.date_time)
    df_time.time = Dates.value.(Dates.Hour.(weather_df.date_time))
    df_time.is_holiday = [date in holiday_dates ? 1 : 0 for date in df_time.date_time]

    # merge weather data
    df_feat = DataFrame()
    df_feat = unique(rightjoin(df_time, weather_df, on=:date_time))


    # select DMA to analyze and merge feature and inflow dataframes
    dma_id = "dma_c"
    df = DataFrame()
    df = leftjoin(inflow_df[!, ["date_time", dma_id]], df_feat, on=:date_time)

    # data cleanup
    delete!(df, [7274, 7277]) # delete duplicate data from autumn time change
    rename!(df, Dict(dma_id => :dma_inflow))
    dropmissing!(df)

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

    # print(IAI.score(grid, train_X, train_y, criterion=:mse))
    print(IAI.score(grid, test_X, test_y, criterion=:mse))

    predict_y

    # # comparison plot
    time = 1:length(test_y)
    plt = plot(time, test_y, label="Actual")
    plt = plot!(time, predict_y, label="Predict")

    plt = xlabel!("Time")
    plt = ylabel!("Value")
    plt = title!("Time Series Plot")

    # Show the plot
    display(plt)

    # write results to csv file
    # results_df = DataFrame(Time=time, Series1=test_y, Series2=predict_y)
    # CSV.write(data_path * "results.csv", results_df)
end