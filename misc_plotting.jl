# some plots for the conference presentation
using Plots
using Measures
using Colors

default_palette = theme_palette(:auto);

dma_labels = ["A - Hospital", "B - Residential (countryside)", "C - Residential (countryside)", "D - Residential/Commercial", "E - Residential/Commercial", "F - Mixed (suburban)", "G - Residential (urban)", "H - Mixed (urban)", "I - Industrial/Commercial", "J - Industrial/Commercial"]
dma_connections = [162, 531, 607, 2094, 7955, 1135, 3180, 2901, 425, 776]
dma_amf = [8.4, 9.6, 4.3, 32.9, 78.3, 8.1, 25.1, 20.8, 20.6, 26.4]

# bar plot
fig = bar(dma_labels, dma_amf, color=default_palette[4], legend=false, ylabel="Average net inflow [L/s]", xrotation=45, top_margin=5mm, bottom_margin=10mm, left_margin=5mm)