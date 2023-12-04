### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 92aa4e68-0eec-4ae3-b315-664f3be97bf7
begin
	import Pkg
	Pkg.activate("/home/bradw/.julia/environments/Julia_v1_9_3__IAI_v3_2_0")
end

# ╔═╡ 7810ca05-3e5e-4858-892c-e5d810154be3
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

# ╔═╡ 0d2c8e2e-906b-11ee-3547-79c59b3009cb
md"""
# Short-term water demand forecasting using optimal regression trees
"""

# ╔═╡ 44762020-2344-47c1-b50c-e163eba20c3e
md"""
Authors: Bradley Jenks, Carlos Jara Arriagada, Yuanyang Liu, and Aly-Joy Ulusoy\
Date: December 2023
"""

# ╔═╡ 8f972901-0762-42f2-8491-c35d901ea920
md"""
This notebook applies algorithms developed by **Interpretable AI** to forecast short-term water demand for the operation of water supply networks. Time series of inflow and weather data were provided by the Battle of Water Networks competition for the **3rd Joint WDSA/CCWI International Conference** from 01--04 July 2024 in Ferrara, Italy.
"""

# ╔═╡ 4e15dca1-1d70-469b-af49-7b4a3be31cec
md"""
### Preamble
"""

# ╔═╡ be3d9574-4e7f-4602-8115-bb9a4b49a7e1
md"""
Activate Julia version with Interpretable AI installation.
"""

# ╔═╡ e2eea5a8-574d-47ab-a5cc-6e1b687abbe9
md"""
Load dependencies and functions located in project environment.
"""

# ╔═╡ 1e06af59-c9d6-4de9-a210-256d22c3faf8
md"""
Load time series data from CSV files.
"""

# ╔═╡ eb04e35e-8f9a-4b31-8669-b62be4688705
begin
	data_path = "/home/bradw/workspace/water_demand_forecasting/data/"
	inflow_df = load_data(data_path, "inflow")
	weather_df = load_data(data_path, "weather")
end

# ╔═╡ f814b428-a19e-46e0-8a12-33c64f482098
md"""
Plot time series data.
"""

# ╔═╡ 9a32fa1e-1751-43b9-ac1e-5f32751446b8
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
	
	@df plot_df plot(:date_time, cols(col_names), ylabel=y_label, palette=:Set1_5, size=(700, 400))
end

# ╔═╡ 20d72099-5e50-4149-b263-32e20eec5772
md"""
### Impute missing data and make dataframe"""

# ╔═╡ c7682d4d-306d-48e7-99c9-3e99dd9b1427
md"""
Preprocessing step to fill in missing values for `inflow_df` and `weather_df` datasets. We investigate four data imputation methods:
- mean
- k-nearest neighbors
- support vector machine
- classification and regression trees
These methods are implemented using Interpretable AI's `opt.impute` algorithm (insert citation here).
"""

# ╔═╡ 9c371dd8-ab51-4efa-baf0-3f94ce4728f8
md"""
We first check the percentage of data with missing values.
"""

# ╔═╡ bbf74005-2cc3-482c-b4c5-331ac6d4035f
begin
	df = inflow_df # inflow_df, weather_df
	DataFrame(col=propertynames(df),
	          missing_fraction=[mean(ismissing.(col)) for col in eachcol(df)])
end

# ╔═╡ 14f0afa3-af20-422d-980d-9187befb29ac
# select data imputation method and call function to create full dataframe
lnr = IAI.ImputationLearner(method=:opt_knn, random_seed=1)

# ╔═╡ ae8a36ed-8225-4d40-a707-08c3ea601c7f
# insert plotting code to view original and imputated time series

# ╔═╡ 8c0b3d5d-17ed-4851-8385-280cbeb87dbf
# call function to save imputed data to csv file

# ╔═╡ d59861ad-42c0-47ae-9f78-e2670df956d1


# ╔═╡ 12a5f040-8346-4c86-ba0c-d7b780dd669f
md"""
### Run optimal regression trees algorithm
"""

# ╔═╡ 3741e18f-708a-4dbc-8533-896416eb0658
md"""
Enter text here on optimal regression trees algorithm and reference Interpretable AI paper and algorithms...
"""

# ╔═╡ cf0f0425-99df-4535-a30c-2b4fd0e259a4
# call function to split full dataframe into train and test datasets (nb: this should include a variable to decide how much historical data is used in the training dataset)

# ╔═╡ 3e175bac-e93b-41cc-8a2c-a4be6efb3be2
# call function to train optimal regression tree model

# ╔═╡ 15aef186-63a2-4880-9507-83889f19c830
# inert code to plot the resulting regression tree

# ╔═╡ 84412d3e-f91a-4c4d-a7b9-6621df03c70e
# call function to generate results for test dataset

# ╔═╡ 7267d3c1-6fa5-48b6-aeaa-e4f1aa7d5eef
# insert plotting code to visualise predicted v. actual demand over test period

# ╔═╡ 6f457dba-1773-4261-b14d-4983bbf5af46
# call code to compute forecasting performance metrics

# ╔═╡ 1d44a908-ff58-4071-b1be-3b2870026327
# call function to save demand forecast to csv file

# ╔═╡ 5defbdb9-4d4b-4767-a687-1ca37b5bfff2
md"""
### Comparison of demand forecasting methods
"""

# ╔═╡ c3cdb40a-d87f-4d9b-bb59-21875003bbe8
md"""
Load demand forecasting results from the different methods tested and compare performance. Which method wins!?
"""

# ╔═╡ Cell order:
# ╟─0d2c8e2e-906b-11ee-3547-79c59b3009cb
# ╟─44762020-2344-47c1-b50c-e163eba20c3e
# ╟─8f972901-0762-42f2-8491-c35d901ea920
# ╟─4e15dca1-1d70-469b-af49-7b4a3be31cec
# ╟─be3d9574-4e7f-4602-8115-bb9a4b49a7e1
# ╠═92aa4e68-0eec-4ae3-b315-664f3be97bf7
# ╠═e2eea5a8-574d-47ab-a5cc-6e1b687abbe9
# ╠═7810ca05-3e5e-4858-892c-e5d810154be3
# ╠═1e06af59-c9d6-4de9-a210-256d22c3faf8
# ╠═eb04e35e-8f9a-4b31-8669-b62be4688705
# ╠═f814b428-a19e-46e0-8a12-33c64f482098
# ╠═9a32fa1e-1751-43b9-ac1e-5f32751446b8
# ╠═20d72099-5e50-4149-b263-32e20eec5772
# ╠═c7682d4d-306d-48e7-99c9-3e99dd9b1427
# ╠═9c371dd8-ab51-4efa-baf0-3f94ce4728f8
# ╠═bbf74005-2cc3-482c-b4c5-331ac6d4035f
# ╠═14f0afa3-af20-422d-980d-9187befb29ac
# ╠═ae8a36ed-8225-4d40-a707-08c3ea601c7f
# ╠═8c0b3d5d-17ed-4851-8385-280cbeb87dbf
# ╠═d59861ad-42c0-47ae-9f78-e2670df956d1
# ╟─12a5f040-8346-4c86-ba0c-d7b780dd669f
# ╟─3741e18f-708a-4dbc-8533-896416eb0658
# ╠═cf0f0425-99df-4535-a30c-2b4fd0e259a4
# ╠═3e175bac-e93b-41cc-8a2c-a4be6efb3be2
# ╠═15aef186-63a2-4880-9507-83889f19c830
# ╠═84412d3e-f91a-4c4d-a7b9-6621df03c70e
# ╠═7267d3c1-6fa5-48b6-aeaa-e4f1aa7d5eef
# ╠═6f457dba-1773-4261-b14d-4983bbf5af46
# ╠═1d44a908-ff58-4071-b1be-3b2870026327
# ╟─5defbdb9-4d4b-4767-a687-1ca37b5bfff2
# ╟─c3cdb40a-d87f-4d9b-bb59-21875003bbe8
