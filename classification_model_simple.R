############################################
## Author: Quinn H. Adams
## code accompanying: Evaluating the Contribution of Weather Variables to Machine Learning Forecasts of Visceral Leishmaniasis in Brazil
## XGBoost classification model (22 select municipalities) for risk forecasting with sliding window
############################################

library(xgboost)
library(caret)
library(ggplot2)
library(dplyr)
library(sf)
library(zoo)
library(data.table)
library(SHAPforxgboost)
library(shapviz)
library(tidyr)
library(lightgbm)


# Load the data
data <- read.csv(paste0(outdir2, "final_data.csv")) %>%
  dplyr::select(!c(X, Sim, Total, municipioname)) %>%
  dplyr::rename(cases = count)

#select 22 municipalites
data <- data %>%
  dplyr::filter(property_id =="150210" | property_id == "150215" | property_id == "150270" | property_id =="150295" 
                | property_id =="150420" | property_id =="150553" | property_id =="150613" | property_id =="170210"
                | property_id =="172100"| property_id =="210160" | property_id =="210300" | property_id =="210320"
                | property_id =="210330" | property_id =="210530" | property_id =="211130" | property_id =="221100"
                | property_id =="230370" | property_id =="231290" | property_id =="280030" | property_id =="313130"
                | property_id =="314330"  | property_id =="350600")
data <- as.data.table(data)
data$population <- as.numeric(data$population)

# Ensure the dataset has a date column
data$date <- as.Date(data$date)
data <- data[order(property_id, date)]

#calculate population density
municipalities <- read_sf(paste0(dir, "municipios.shp")) %>%
  dplyr::rename(property_id = CD_MUN)

municipalities <- municipalities[!duplicated(municipalities$property_id), ]
municipalities$property_id <- substr(municipalities$property_id, start = 1, stop = 6)
municipalities$area <- st_area(municipalities) /1000000

municipalities$area <- as.numeric(municipalities$area)

municipalities <- municipalities %>%
  dplyr::select(property_id, area)

municipalities$property_id <- as.integer(municipalities$property_id)
data <- left_join(data, municipalities)
data$pop_density <- data$population/data$area

# Calculate the moving average
data <- data %>%
  group_by(property_id) %>%
  mutate(moving_avg = zoo::rollmean(cases, k = 36, fill = NA, align = "right"))


data$binary_risk <- ifelse(data$moving_avg > 2.4, 1, 0)

# Parameters
window_size <- 24 # Sliding window size (in months)
forecast_horizon <- 3 # Forecast horizon (in months)

data <- as.data.table(data)


# Generate lagged weather variables (up to 4 months lag)
weather_vars <- c("temp_C", "RH", "precip_mm", "dpt", "svpd", "windspeed", "ONI")
for (var in weather_vars) {
  for (lag in 1:6) {
    lagged_var_name <- paste0(var, "_lag", lag)
    data[, (lagged_var_name) := data.table::shift(get(var), n = lag, type = "lag"), by = property_id]
  }
}

# Prepare predictors and response variables
max_case_lag <- 6 # Include lagged cases up to 12 months
for (lag in 1:max_case_lag) {
  lagged_case_name <- paste0("lag_cases_", lag)
  data[, (lagged_case_name) := shift(cases, n = lag, type = "lag"), by = property_id]
}

# Handle categorical variables
categorical_vars <- c("social_prosperity",  "slt", "koppen", "month", "region")
dummies <- dummyVars(~ ., data = data[, ..categorical_vars], fullRank = TRUE)
encoded_data <- data.table(predict(dummies, newdata = data))

data <- cbind(data[, !categorical_vars, with = FALSE], encoded_data)

# Standardize non-time-varying features using global statistics
altitude_mean <- mean(unique(data$altitude))
altitude_sd <- sd(unique(data$altitude))

svi_mean <- mean(unique(data$SVI))
svi_sd <- sd(unique(data$SVI))

mhdi_mean <- mean(unique(data$MHDI))
mhdi_sd <- sd(unique(data$MHDI))

gini_mean <- mean(unique(data$gini_index))
gini_sd <- sd(unique(data$gini_index))

income_mean <- mean(unique(data$percentwithoutincome))
income_sd <- sd(unique(data$percentwithoutincome))

data <- data %>%
  mutate(
    altitude = (altitude - altitude_mean) / altitude_sd,
    SVI = (SVI - svi_mean) / svi_sd,
    MHDI = (MHDI - mhdi_mean) / mhdi_sd,
    gini_index = (gini_index - gini_mean) / gini_sd,
    percentwithoutincome = (percentwithoutincome - income_mean)/income_sd
  )

data$temp_svi <- data$temp_C*data$SVI
data$temp_urban <- data$temp_C*data$urban_est
data$temp_mhdi <- data$temp_C*data$MHDI
data$prec_slt <- data$precip_mm*data$slt
data$temp_electric <- data$temp_C*data$electricity
data$oni_footprint <- data$ONI*data$human_footprint_est
data$oni_urban <- data$ONI*data$urban_est
data$oni_rh <- data$ONI*data$RH
data$oni_temp <- data$temp_C*data$ONI
data$oni_precip <- data$precip_mm*data$ONI

# Identify categorical variable names using pattern matching
categorical <- grep("social_prosperity|koppen|month|region", names(data), value = TRUE)

# Remove rows with NAs due to lagging
data <- na.omit(data)

# Split into time-varying and time-invariant predictors
lagged_case_vars <- paste0("lag_cases_", 1:max_case_lag)
time_varying <- c("temp_C", "RH", "precip_mm", "dpt", "svpd", "windspeed", "ONI",
                  paste0("temp_C_lag", 1:6), 
                  paste0("RH_lag", 1:6), 
                  paste0("precip_mm_lag", 1:6), 
                  paste0("dpt_lag", 1:6), 
                  paste0("svpd_lag", 1:6), 
                  paste0("windspeed_lag", 1:6), 
                  paste0("ONI_lag", 1:6),
                  "human_footprint_est", "urban_est", "prop_coinfection", "pop_density",
                  "forest_est", "pasture_est", "cropland_est", "temp_svi", "temp_urban",
                  "temp_mhdi", "prec_slt", "temp_electric", "oni_footprint", "oni_urban", "oni_rh", "oni_temp", "oni_precip",
                  lagged_case_vars)
time_invariant <- c("slt", "water_scarcity", "SVI", "MHDI",  "electricity", "gini_index", "Hsdiploma", "av_income_employed", "percentwithoutincome",  "altitude")


data <- as.data.table(data)


# Sliding window forecast function
sliding_window_forecast <- function(data, window_size, forecast_horizon) {
  municipalities <- unique(data$property_id)
  results <- list()
  models <- list()
  shap_results <- list()
  
  
  for (muni in municipalities) {
    muni_data <- data[property_id == muni]
    
    for (start_idx in 1:(nrow(muni_data) - window_size - forecast_horizon)) {
      train_data <- muni_data[start_idx:(start_idx + window_size - 1)]
      test_data <- muni_data[(start_idx + window_size):(start_idx + window_size + forecast_horizon - 1)]
      
      # Prepare training and testing datasets
      train_x <- as.matrix(cbind(train_data[, ..time_varying], train_data[, ..time_invariant], train_data[, ..categorical]))
      train_y <- train_data$binary_risk
      
    
      train_y <- as.integer(train_y) 
      
      test_x <- as.matrix(cbind(test_data[, ..time_varying], train_data[, ..time_invariant], train_data[, ..categorical]))
      test_y <- test_data$binary_risk
      
      # Train XGBoost model
      dtrain <- xgb.DMatrix(data = train_x, label = train_y)
      params <- list(
        objective = "binary:logistic",
        eta = 0.1,
        max_depth = 4,
        subsample = 0.8,
        colsample_bytree = 0.8
      )
      
      model <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
      
      # Save the model in the models list
      model_key <- paste0("muni_", muni, "_window_", start_idx)
      models[[model_key]] <- model
      
      # Forecast
      preds <- predict(model, xgb.DMatrix(data = test_x))
      

      
      tryCatch({
        # Compute SHAP values
        # 1. Compute SHAP values
        shap_vals <- shap.values(xgb_model = model, X_train = train_x)
        
        # 2. Prepare SHAP data in long format (only includes SHAP values)
        shap_long <- shap.prep(
          xgb_model = model,
          X_train = train_x,
          shap_contrib = shap_vals$shap_score,
          top_n = NULL
        )
        
        train_x_df <- as.data.frame(train_x)
        
        # 3. Add observation ID to track rows
        shap_long <- shap_long %>%
          mutate(ID = rep(1:nrow(train_x), times = ncol(train_x)))  # one ID per row
        
        # 4. Reshape original feature values to long format
        # Reshape feature values to long format
        feature_values_long <- train_x_df %>%
          mutate(ID = 1:nrow(.)) %>%
          pivot_longer(
            cols = -ID,
            names_to = "variable",
            values_to = "feature_value"
          )
        
        
        # 5. Merge SHAP values with actual feature values
        shap_long <- left_join(shap_long, feature_values_long, by = c("ID", "variable"))
        
        # 6. Clean up column names for plotting
        shap_long <- shap_long %>%
          rename(mean_shap = value)
        # Store long-form results (with SHAP + feature values) for plotting
        shap_results[[model_key]] <- shap_long
        
      }, error = function(e) {
        message("SHAP computation failed for ", muni, " at window ", start_idx, " | Error: ", e$message)
      })
      
      
      # Save results
      results[[model_key]] <- data.table(
        property_id = muni,
        date = test_data$date,
        observed = test_y,
        predicted = preds
      )
    }
  }
  
  # Combine forecast results and return both results and models
  forecast_results <- rbindlist(results, fill = TRUE)
  return(list(forecast_results = forecast_results, models = models, shap_results = shap_results))
}


# Run sliding window forecast
forecast_output <- sliding_window_forecast(data, window_size, forecast_horizon)

# Extract forecast results and models
forecast_results <- forecast_output$forecast_results

# Evaluate performance
library(dplyr)
library(pROC)

# Function to compute performance metrics per property_id
evaluate_model_by_property <- function(forecast_results) {
  forecast_results %>%
    group_by(property_id) %>%
    summarise(
      Accuracy = mean(round(predicted) == observed, na.rm = TRUE),
      Precision = sum((round(predicted) == 1) & (observed == 1), na.rm = TRUE) / sum(round(predicted) == 1, na.rm = TRUE),
      Recall = sum((round(predicted) == 1) & (observed == 1), na.rm = TRUE) / sum(observed == 1, na.rm = TRUE),
      F1_Score = 2 * (Precision * Recall) / (Precision + Recall),
      AUC = if (length(unique(cur_data()$observed)) > 1) {
        as.numeric(auc(roc(cur_data()$observed, cur_data()$predicted, quiet = TRUE)))
      } else {
        NA  # Return NA if only one class exists
      }
    ) %>%
    mutate(F1_Score = ifelse(is.na(F1_Score), 0, F1_Score))  # Handle NaN cases
}


# Compute performance by property_id
performance_by_property <- evaluate_model_by_property(forecast_results)

# Print results
print(performance_by_property)

summary(performance_by_property$AUC)
write.csv(performance_by_property, paste0(dir2, "classification_results22.csv"))


#visualize the AUC, precision, and recall
results <- read.csv(paste0(dir2, "classification_results22.csv"))%>%
  dplyr::select(property_id, Precision, Recall, AUC)


Dat.stacked <- results %>% pivot_longer(!c(property_id), names_to = "metric", values_to = "value")
group.colors <- c(AUC = "#440154", Precision = "#33638D", Recall ="#55C667")

ggplot(Dat.stacked, aes(x=metric, y=value, color = metric)) + 
  geom_violin() + 
  geom_boxplot(width=0.1, fill="white")+
  labs(title="",x="", y = "")+
  labs(color='Success Metric') +
  scale_color_manual(values = group.colors)+
  theme_classic()+
  theme(axis.text.x = element_text(size = 14),   # Change x-axis label size
        axis.text.y = element_text(size = 14) ,
        legend.text=element_text(size=14), 
        legend.title = element_text(size=14))  # Change y-axis label size+


ggsave(filename = "classification_violin",
       plot = last_plot(),
       width = 200, # 14.1 x 5.05 in 358 x 256 mm 
       height = 200,# 
       units = "mm",
       dpi = 300,
       device = "pdf"
)


shap_all <- rbindlist(forecast_output$shap_results, use.names = TRUE, fill = TRUE, idcol = "list_name")

# Add a property_id column by extracting the first 6 characters of list_name
shap_all[, property_id := substr(list_name, 6, 11)]

# Optionally, remove the list_name column if it's not needed
# shap_all[, list_name := NULL]

# Ensure you retain the feature names for interpretation
shap_all[, feature := as.character(variable)]



shap_summary <- shap_all[, .(mean_shap = mean(abs(mean_shap)), 
                             sd_shap = sd(abs(mean_shap))), 
                         by = feature, property_id]

# Summarize SHAP values for each feature
shap_summary <- shap_all[, .(mean_shap = mean(abs(mean_shap)), 
                             sd_shap = sd(abs(mean_shap))), 
                         by = variable] %>%
  filter(mean_shap>0)

# Plot the SHAP summary
shap_all_sum <- shap_all %>%
  dplyr::group_by(variable, property_id)%>%
  summarise(mean_shap = (abs(mean_shap)))%>%
  filter(mean_shap != 0)


ggplot(shap_all_sum, aes(x = reorder(variable, abs(mean_shap)), y = abs(mean_shap)))+
  geom_point( color="#440154", alpha=0.6) + 
  coord_flip()+
  theme_minimal()+  
  theme(axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))+
  labs(
    x = "Feature",
    y = "SHAP Value Distribution")

ggsave(filename = "shap_dis_allvars",
       plot = last_plot(),
       width = 220, # 14.1 x 5.05 in 358 x 256 mm 
       height = 260,# 
       units = "mm",
       dpi = 300,
       device = "pdf")


# Define the full list of features from the SHAP plot (excluding lagged variables)
all_features <- c("ONI", "human_footprint_est", "windspeed_lag1", 
                  "temp_urban","urban_est", "svpd_lag6",
                  "dpt_lag4", "temp_C_lag6", "prop_coinfection", 
                  "pop_density", "forest_est", "oni_temp", "temp_svi", 
                  "precip_mm_lag1", 
                  "lag_cases_5", "RH_lag2")

# Create a dictionary of names
rename_dict <- c(
  "ONI" = "Oceanic Niño Index*",
  "human_footprint_est" = "Human Footprint",
  "windspeed_lag1" = "Windspeed (m/s) (lag 1)*",
  "temp_urban" = "Temperature x Urban*",
  "urban_est" = "Urban Coverage",
  "dpt_lag4" = "Dew Point Temperature (°C) (lag 4)*",
  "temp_C_lag6" = "Average Temperature (°C) (lag 6)*",
  "prop_coinfection" = "Proportion with HIV Coinfection",
  # "pasture_est" = "Pasture Coverage",
  "cropland_est" = "Cropland Coverage",
  "pop_density" = "Population Density",
  "forest_est" = "Forest Coverage",
  # "month" = "Month",
  "svpd_lag6" = "Saturation Vapor Pressure Deficit (kPa) (lag 6)*",
  "temp_svi" = "Temperature x SVI*",
  "precip_mm_lag1" = "Total Precipitation (mm) (lag 1)*",
  "lag_cases_5" = "Five Month Lagged Case Count", 
  "RH_lag2" = "Relative Humidity (%) (lag 2)*",
  "oni_temp" = "Oceanic Niño Index x Temperature*"
)
library(stringr)

# Convert names to updated versions
non_lagged_features <- sapply(all_features, function(x) rename_dict[[x]], USE.NAMES = FALSE)


# Print new variable names
print(non_lagged_features)

filtered_shap <- shap_all_sum %>%
  dplyr::filter(variable %in% c("ONI", "human_footprint_est", "windspeed_lag1", 
                                "temp_urban","urban_est", "svpd_lag6",
                                "dpt_lag4", "temp_C_lag6", "prop_coinfection", 
                                "pop_density", "forest_est", "oni_temp", "temp_svi", 
                                "precip_mm_lag1", 
                                "lag_cases_5", "RH_lag2"))

filtered_shap <- filtered_shap %>%
  mutate(variable = recode(variable, !!!rename_dict)) 


ggplot(filtered_shap, aes(x = reorder(variable, abs(mean_shap)), y = abs(mean_shap)))+
  geom_point(notch=FALSE, outlier.shape = NULL, color="#440154", alpha=0.6) + 
  coord_flip()+
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))+
  # ylim(-0.3,0.3)+
  labs(
    x = "Feature",
    y = "SHAP Value Distribution")

ggsave(filename = "shap_dist_nonlagged_vars",
       plot = last_plot(),
       width = 200, # 14.1 x 5.05 in 358 x 256 mm 
       height = 200,# 
       units = "mm",
       dpi = 300,
       device = "pdf")


#plot beeswarm per muni: 
library(shapviz)
library(ggplot2)
install.packages("shapper")
install.packages("ggbeeswarm")
library(ggbeeswarm)
library(shapper)
library(ggpubr)


generate_shap_plots_for_municipality <- function(shap_all, municipality_id) {
  # Ensure property_id exists
  if (!"property_id" %in% colnames(shap_all)) {
    stop("Error: 'property_id' column is not present in shap_data.")
  }
  
  # Filter for the specified municipality
  muni_shap_data <- shap_all[property_id == municipality_id]
  if (nrow(muni_shap_data) == 0) {
    stop("Error: No data found for the specified municipality.")
  }
  
  # Extract unique feature names
  feature_names <- unique(muni_shap_data$variable)
  
  
  # Calculate the mean absolute SHAP value for reordering
  variable_order <- muni_shap_data[, .(mean_abs_shap = mean(abs(mean_shap), na.rm = TRUE)), by = variable]
  variable_order <- variable_order[order(-mean_abs_shap)]  # Order by descending mean_abs_shap
  
  # Select top 15 variables
  top_variables <- variable_order$variable[1:min(15, nrow(variable_order))]
  
  # Filter data to only top 15 variables
  muni_shap_data <- muni_shap_data[variable %in% top_variables]
  
  
  # Add factor levels to `variable` with reversed order (highest on top)
  muni_shap_data[, variable := factor(variable, levels = rev(variable_order$variable))]
  
  #creating a scaled feature value
  muni_shap_data <- muni_shap_data %>%
    group_by(variable) %>%
    mutate(
      scaled_feature_value = (feature_value - min(feature_value, na.rm = TRUE)) /
        (max(feature_value, na.rm = TRUE) - min(feature_value, na.rm = TRUE))
    ) %>%
    ungroup()
  
  muni_shap_data <- muni_shap_data %>%
    group_by(variable)%>%
    filter(sum(mean_shap) != 0 )%>%
    ungroup()
  # Beeswarm Plot
  beeswarm_plot <- ggplot(muni_shap_data, aes(x = mean_shap, y = variable)) +
    geom_quasirandom(aes(color = scaled_feature_value), alpha = 0.7, width = 0.2) +
    scale_color_viridis_c(
      option = "viridis",
      name = "Feature Value",
      breaks = c(min(muni_shap_data$scaled_feature_value, na.rm = TRUE), max(muni_shap_data$scaled_feature_value, na.rm = TRUE)),
      labels = c("Low", "High")
    ) +
    labs(
      title = "Aracaju, Sergipe",
      x = "SHAP Value",
      y = "Feature"
    ) +
    theme_minimal() +
    theme(legend.title = element_text(size = 12),
          legend.text = element_text(size = 12),
          plot.title = element_text(face="bold",hjust = 0.5),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12))
  
  
  # Print the plot
  print(beeswarm_plot)
}


p1 <-  generate_shap_plots_for_municipality(shap_all, municipality_id = "350600")
p2 <-  generate_shap_plots_for_municipality(shap_all, municipality_id = "170210")
p3 <-  generate_shap_plots_for_municipality(shap_all, municipality_id = "280030")
p4 <-  generate_shap_plots_for_municipality(shap_all, municipality_id = "211130")

#arrange plot
ggarrange(p1, p2, p3, p4)

#save
ggsave(filename = "figure4",
       plot = last_plot(),
       width = 300, # 14.1 x 5.05 in 358 x 256 mm 
       height = 250,# 
       units = "mm",
       dpi = 400,
       device = "pdf"
)

