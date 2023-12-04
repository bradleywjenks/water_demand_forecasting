using CSV
using DataFrames
using Dates


function read_data(data_path::String, data_type::String)

    if data_type == "inflow"
        df = CSV.read(data_path * "InflowData_1.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    elseif data_type == "weather"
        df = CSV.read(data_path * "WeatherData_1.csv", DataFrame)
        df.date_time = Dates.DateTime.(df.date_time, "dd/mm/yyyy HH:MM")

    end

    return df
end


function write_data(data_path::String, data_type::String, df)

    if data_type == "inflow"
        CSV.write(data_path * "inflow_data_imputed.csv", df)

    elseif data_type == "weather"
        CSV.write(data_path * "weather_data_imputed.csv", df)

    end

end


function make_dataframe(inflow_df, weather_df, lag_times, dma_id)

    # define holiday dates
    holiday_dates = [Dates.Date("2021-01-01"), Dates.Date("2021-01-06"), Dates.Date("2021-04-04"), Dates.Date("2021-04-05"), Dates.Date("2021-04-25"), Dates.Date("2021-05-01"), Dates.Date("2021-06-02"), Dates.Date("2021-08-15"), Dates.Date("2021-11-01"), Dates.Date("2021-11-03"), Dates.Date("2021-12-08"), Dates.Date("2021-12-25"), Dates.Date("2021-12-26"), Dates.Date("2022-01-01"), Dates.Date("2022-01-06"), Dates.Date("2022-04-17"), Dates.Date("2021-04-18"), Dates.Date("2022-04-25"), Dates.Date("2022-05-01"), Dates.Date("2022-06-02"), Dates.Date("2022-08-15")]

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
    df = DataFrame()
    df = leftjoin(inflow_df[!, [:date_time, dma_id]], df_feat, on=:date_time)

    # data cleanup
    delete!(df, [7274, 7277]) # delete duplicate data from autumn time change
    rename!(df, Dict(dma_id => :dma_inflow))

    # lagged values
    for (i, v) ∈ enumerate(lag_times)
        df[!, Symbol("prev_", v, "_inflow")] = [fill(missing, v); df.dma_inflow[1:end-v]]
    end

    return df
    
end

