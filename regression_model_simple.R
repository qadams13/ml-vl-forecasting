############################################
## Author: Quinn H. Adams
## code accompanying: Evaluating the Contribution of Weather Variables to Machine Learning Forecasts of Visceral Leishmaniasis in Brazil
## XGBoost regression model (113 municipalities) for case count forecasting with sliding window
############################################

#libraries
library(xgboost)
library(dplyr)
library(data.table)
library(caret)
library(Metrics)
library(sf)
library(SHAPforxgboost)
library(shapviz)
library(ggpubr)
library(doParallel)
library(foreach)
library(data.table)
library(future)
library(future.apply)

#read in dataset 
#these data include only the 113 selected municipalities (based on a window threshold of 10 cases per window to ensure 
#enough data for training the model)

data <- read.csv(paste0(outdir2, "final_data.csv"))

# Parameters
window_size <- 24 # Sliding window size (in months)
forecast_horizon <- 3 # Forecast horizon (in months)

# Generate lagged weather variables (up to 4 months lag)
weather_vars <- c("temp_C", "RH", "precip_mm", "dpt", "SH", "svp", "svpd", "windspeed")
for (var in weather_vars) {
  for (lag in 1:6) {
    lagged_var_name <- paste0(var, "_lag", lag)
    data[, (lagged_var_name) := data.table::shift(get(var), n = lag, type = "lead"), by = property_id]
  }
}

# Prepare predictors and response variables
max_case_lag <- 12 # Include lagged cases up to 12 months
for (lag in 1:max_case_lag) {
  lagged_case_name <- paste0("lag_cases_", lag)
  data[, (lagged_case_name) := shift(cases, n = lag, type = "lag"), by = property_id]
}

# Handle categorical variables
categorical_vars <- c("social_prosperity",  "slt", "koppen", "month", "region")
dummies <- dummyVars(~ ., data = data[, ..categorical_vars], fullRank = TRUE)
encoded_data <- data.table(predict(dummies, newdata = data))
data <- cbind(data[, !categorical_vars, with = FALSE], encoded_data)

# Remove rows with NAs due to lagging
data <- na.omit(data)



# Split into time-varying and time-invariant predictors
lagged_case_vars <- paste0("lag_cases_", 1:max_case_lag)

time_varying <- c("temp_C", "RH", "precip_mm", "dpt", "SH", "svp", "svpd", "windspeed",
                  paste0("temp_C_lag", 1:6), 
                  paste0("RH_lag", 1:6), 
                  paste0("precip_mm_lag", 1:6), 
                  paste0("dpt_lag", 1:6), 
                  paste0("svpd_lag", 1:6), 
                  paste0("windspeed_lag", 1:6), 
                  "human_footprint_est", "urban_est", "ONI", "prop_coinfection", "temp_svi", "temp_urban",
                  lagged_case_vars)

time_invariant <- c("slt", "water_scarcity", "SVI", "MHDI",  "electricity", "gini_index", "Hsdiploma", "av_income_employed", "percentwithoutincome",  "altitude")
categorical <- grep("social_prosperity|koppen|month|region", names(data), value = TRUE)


data <- as.data.table(data)


sliding_window_forecast <- function(data, window_size, forecast_horizon, time_varying, time_invariant, n_bootstrap = 100) {
  # Ensure data is a data.table
  if (!is.data.table(data)) {
    data <- as.data.table(data)
  }
  
  # Check that property_id exists
  if (!"property_id" %in% colnames(data)) {
    stop("Column 'property_id' is missing from the dataset")
  }
  
  municipalities <- unique(data$property_id)
  
  # Function to process a single municipality
  process_municipality <- function(muni, data, time_varying, time_invariant) {
    # Filter data for the municipality
    muni_data <- data[property_id == muni, ]
    
    results <- list()
    shap_results <- list()
    models <- list()
    
    for (start_idx in 1:(nrow(muni_data) - window_size - forecast_horizon)) {
      train_data <- muni_data[start_idx:(start_idx + window_size - 1), ]
      test_data <- muni_data[(start_idx + window_size):(start_idx + window_size + forecast_horizon - 1), ]
      
      # Prepare training and testing datasets
      train_x <- as.matrix(cbind(train_data[, ..time_varying, with = FALSE], train_data[, ..time_invariant, with = FALSE]))
      test_x <- as.matrix(cbind(test_data[, ..time_varying, with = FALSE], test_data[, ..time_invariant, with = FALSE]))
      

      train_y <- train_data$cases
      test_y <- test_data$cases
      
      # Initialize bootstrap predictions
      bootstrap_preds <- matrix(0, nrow = n_bootstrap, ncol = nrow(test_x))
      
      for (b in 1:n_bootstrap) {
        # Bootstrap sample with replacement
        bootstrap_indices <- sample(seq_len(nrow(train_x)), size = nrow(train_x), replace = TRUE)
        bootstrap_x <- train_x[bootstrap_indices, , drop = FALSE]
        bootstrap_y <- train_y[bootstrap_indices]
        
        # Train XGBoost model
        dtrain <- xgb.DMatrix(data = bootstrap_x, label = bootstrap_y)
        params <- list(
          objective = "reg:squarederror",
          eta = 0.1,
          max_depth = 4,
          subsample = 0.8,
          colsample_bytree = 0.8,
          min_child_weight = 7
        )
        
        model <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
        bootstrap_preds[b, ] <- predict(model, xgb.DMatrix(data = test_x))
        
        # Save model
        model_key <- paste0("muni_", muni, "_window_", start_idx, "_bootstrap_", b)
        models[[model_key]] <- model
      }
      
      # Aggregate predictions
      mean_preds <- rowMeans(bootstrap_preds)
      median_preds <- apply(bootstrap_preds, 2, median)
      lower_ci <- apply(bootstrap_preds, 2, quantile, probs = 0.025)
      upper_ci <- apply(bootstrap_preds, 2, quantile, probs = 0.975)
      
      # Save results
      results[[paste0("muni_", muni, "_window_", start_idx)]] <- data.table(
        property_id = muni,
        date = test_data$date,
        observed = test_y,
        predicted_mean = mean_preds,
        predicted_lower_ci = lower_ci,
        predicted_upper_ci = upper_ci
      )
    }
    
    list(forecast_results = rbindlist(results, fill = TRUE))
  }
  
  # Parallel processing
  plan(multisession, workers = availableCores() - 1)
  all_results <- future_lapply(
    municipalities,
    function(muni) process_municipality(muni, data, time_varying, time_invariant)
  )
  plan(sequential)  # Reset parallel plan
  
  # Combine results
  forecast_results <- rbindlist(lapply(all_results, `[[`, "forecast_results"), fill = TRUE)
  
  return(forecast_results)
}


#  call sliding window
forecast_output2 <- sliding_window_forecast(
  data = data, 
  window_size = 24, 
  forecast_horizon = 3, #change to 6 or 12 for longer forecast horizons
  time_varying = time_varying, 
  time_invariant = time_invariant
)

#save forecast results
write.csv(forecast_output2, "full113_forecast_output.csv")

#select properties for plotting
municipio_names <- c(
  "211130" = "São Luis, Maranhão",
  "221100" = "Teresina, Piauí",
  "230440" = "Fortaleza, Ceará",
  "310620" = "Belo Horizonte, Minas Gerais",
  "500270" = "Campo Grande, Mato Grosso do Sul"
)

# plot representative properties
plot <- deduplicated_results %>%
  filter(property_id =="210120" | property_id == "291080" |  property_id =="150020" | property_id =="280030")
ggplot(plot, aes(x = date)) +
  # geom_line(aes(y = null_pred, color = "Null"), size = 0.5, linetype = "twodash")+
  geom_line(aes(y = original_observed, color = "Observed"),  size = 0.75, linetype = "twodash") +
  geom_line(aes(y = predicted_mean, color = "Predicted"),  size = 1 ) +
  # geom_line(aes(y = 2.4), linetype = "dotted")+
  geom_ribbon(aes(ymin = predicted_lower_ci, ymax = predicted_upper_ci), fill = "#20A486", alpha = 0.2) +
  facet_wrap(property_id~.,  labeller = as_labeller(municipio_names)) +
  scale_color_manual(values = c("Observed" = "#440154", "Predicted" = "#20A486", "Null" = "#8fd744"))+
  labs(
    x = "Date",
    y = "Cases",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(filename = "timeseries_forecasts",
       plot = last_plot(),
       width = 220, # 14.1 x 5.05 in 358 x 256 mm 
       height = 200,# 
       units = "mm",
       dpi = 300,
       device = "pdf"
)

#calculate percent coverage: 

is_within_range <- function(original_observed, predicted_lower_ci, predicted_upper_ci) {
  original_observed >= predicted_lower_ci & original_observed <= predicted_upper_ci
}

percent <- deduplicated_results %>%
  dplyr::group_by(property_id) %>%
  dplyr::mutate(
    within_range = is_within_range(original_observed, predicted_lower_ci, predicted_upper_ci),
    percentage = round(mean(within_range) * 100, 2)
  )


df_filtered <- deduplicated_results %>%
  dplyr::group_by(property_id)%>%
  filter(original_observed >= predicted_lower_ci & original_observed <= predicted_upper_ci)
percentage <- nrow(df_filtered) / nrow(deduplicated_results) * 100
summary(percent$percentage)

percent <- percent %>%
  group_by(property_id)%>%
  summarise(percent = max(percentage))

