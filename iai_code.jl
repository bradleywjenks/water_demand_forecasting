ENV["JULIA_LOAD_PATH"] = "@:@Julia_v1_9_3__IAI_v3_2_0:@v#.#:@stdlib"
ENV["JULIA_DEPOT_PATH"] = "~/.julia:/home/bradw/.julia/artifacts/5f4ef55ac93b05a932a9f23506f5dbce9f3fe0e8:/home/bradw/.julia/juliaup/julia-1.9.3+0.x64.linux.gnu/local/share/julia:/home/bradw/.julia/juliaup/julia-1.9.3+0.x64.linux.gnu/share/julia"

using CSV
using IJulia
using DataFrames
using Dates

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
    df_time.time = Dates.Time.(weather_df.date_time)
    df_time.is_holiday = [date in holiday_dates ? 1 : 0 for date in df_time.date_time]

    # merge weather data
    df_feat = DataFrame()
    df_feat = unique(rightjoin(df_time, weather_df, on=:date_time))


    # select DMA to analyze and merge feature and inflow dataframes
    dma_id = "dma_a"

    df = DataFrame()
    df = leftjoin(inflow_df[!, ["date_time", dma_id]], df_feat, on=:date_time)
    delete!(df, [7274, 7277]) # delete duplicate data from autumn time change
    rename!(df, Dict(dma_id => :dma_inflow))
end