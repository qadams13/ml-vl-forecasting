############################################
## Author: Quinn H. Adams
## code accompanying: Evaluating the Contribution of Weather Variables to Machine Learning Forecasts of Visceral Leishmaniasis in Brazil
############################################

#risk subset dependence plots for supplement

generate_shap_plots_for_municipality <- function(shap_all, municipality_id, top_n = 15, muni_label = NULL) {
  require(dplyr)
  require(data.table)
  require(ggplot2)
  require(ggbeeswarm)
  require(stringr)
  require(tidyr)
  require(patchwork)  
  require(forcats)
  require(scales)
  require(viridisLite)
  
  # --- checks
  if (!"property_id" %in% names(shap_all)) stop("Error: 'property_id' column is not present in shap_all.")
  if (!all(c("variable","feature_value") %in% names(shap_all))) stop("Need columns: variable, feature_value.")
  if (!("mean_shap" %in% names(shap_all))) stop("Need a SHAP column named 'mean_shap' (one row per observation).")
  
  # --- filter to municipality
  dt <- as.data.table(shap_all)[property_id == municipality_id]
  if (nrow(dt) == 0) stop("Error: No data found for the specified municipality.")
  if (is.null(muni_label)) muni_label <- paste0("Municipality ", municipality_id)
  
  # --- pick top features by mean |SHAP|
  variable_order <- dt[, .(mean_abs_shap = mean(abs(mean_shap), na.rm = TRUE)), by = variable][order(-mean_abs_shap)]
  top_vars <- head(variable_order$variable, min(top_n, nrow(variable_order)))
  
  dt_top <- dt[variable %in% top_vars]
  
  # --- per-feature color scaling (0..1) + flag constant features
  rng <- dt_top[, .(
    vmin  = suppressWarnings(min(feature_value, na.rm = TRUE)),
    vmax  = suppressWarnings(max(feature_value, na.rm = TRUE))
  ), by = variable]
  rng[, range0 := !(is.finite(vmin) & is.finite(vmax)) | (vmax - vmin == 0)]
  
  dt_top <- merge(dt_top, rng, by = "variable", all.x = TRUE, sort = FALSE)
  
  dt_top[, scaled_feature_value :=
           fifelse(range0, 0.5, (feature_value - vmin) / pmax(vmax - vmin, .Machine$double.eps))]
  
  dt_top[, feature_note := fifelse(range0, "(constant value)", "")]
  any_constant <- any(dt_top$range0, na.rm = TRUE)
  
  # order variables (highest influence at top)
  ord_levels <- rev(variable_order[variable %in% top_vars, variable])
  dt_top[, variable := factor(variable, levels = ord_levels)]
  
  # --- BEESWARM (per-feature scaled colors 0..1)
  beeswarm_plot <- ggplot(dt_top, aes(x = mean_shap, y = variable)) +
    ggbeeswarm::geom_quasirandom(
      aes(color = scaled_feature_value),
      alpha = 0.75, width = 0.22, varwidth = FALSE, groupOnX = FALSE, size = 1.3
    ) +
    scale_color_viridis_c(
      option = "viridis", limits = c(0,1),
      name = "Feature value",
      breaks = c(0, 1), labels = c("Low", "High")
    ) +
    labs(
      title ="Araguaína, Tocantins",
      subtitle = if (any_constant)
        "Note: Features marked '(constant value)' have no variation; color fixed at mid-scale."
      else NULL,
      x = "SHAP value (impact on model output)",
      y = "Feature"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      axis.title.x = element_text(margin = margin(t = 6)),
      axis.title.y = element_text(margin = margin(r = 6))
    )
  
  # --- DEPENDENCE SMALL-MULTIPLES (supplementary figure)
  dep_data <- dt_top %>%
    as.data.frame() %>%
    mutate(
      var_label = ifelse(feature_note == "(constant value)",
                         paste0(as.character(variable), " ", feature_note),
                         as.character(variable))
    )
  
  dependence_plot <- ggplot(dep_data, aes(x = feature_value, y = mean_shap)) +
    geom_point(aes(color = scaled_feature_value), alpha = 0.6, size = 0.8) +
    geom_smooth(method = "loess", formula = y ~ x, se = TRUE, span = 0.9, linewidth = 0.6) +
    scale_color_viridis_c(
      option = "viridis", limits = c(0,1),
      name = "Within-feature\nscaled value",
      breaks = c(0, 1), labels = c("Low", "High")
    ) +
    facet_wrap(~ fct_relevel(var_label, rev(unique(var_label))), scales = "free",ncol = 3) +
    labs(
      title = paste0("São Luís, Maranhão: Per-Predictor SHAP Dependence"),
      x = "Feature value (native units)",
      y = "SHAP value"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      strip.text = element_text(size = 9)
    )
  
  # --- return both 
  print(beeswarm_plot)
  print(dependence_plot)
  
  invisible(list(beeswarm = beeswarm_plot, dependence = dependence_plot,
                 top_variables = top_vars,
                 constant_features = unique(dep_data$variable[dep_data$feature_note == "(constant value)"])))
}



generate_shap_plots_for_municipality(shap_all, municipality_id = "211130")

ggsave(filename = "bauru_dependence",
       plot = last_plot(),
       width = 350, # 14.1 x 5.05 in 358 x 256 mm 
       height = 400,# 
       units = "mm",
       dpi = 400,
       device = "png"
)

