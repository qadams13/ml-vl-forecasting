############################################
## Author: Quinn H. Adams
## code accompanying: Evaluating the Contribution of Weather Variables to Machine Learning Forecasts of Visceral Leishmaniasis in Brazil
############################################

#correltation plot between classification and regression models
results_class <- read.csv(paste0(dir2, "classification_results22.csv"))%>%
  dplyr::select(property_id, Precision, Recall, AUC)

results_reg <- read.csv(paste0(dir2, "percent_22muni.csv"))


#merge
perf_df <- left_join(results_class, results_reg)

# Spearman correlation
cor <- cor.test(perf_df$AUC, perf_df$percent, method = "spearman")


library(ggpubr)
ggplot(perf_df, aes(x = AUC, y = percent)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  stat_cor(method = "spearman", label.x = 0.7, label.y = 90) +  # adjust placement as needed
  labs(
    x = "AUC (Classification Performance)",
    y = "95% Prediction Interval Coverage (Regression)",
    title = "Correlation Between Classification and Regression Performance"
  ) +
  theme_minimal()


perf_df <- perf_df %>%
  mutate(
    class_perf = AUC,
    reg_perf = percent,
    perf_diff = scale(class_perf) - scale(reg_perf)
  )


# Normalize to 0-1 range
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
perf_df$norm_auc <- normalize(perf_df$AUC)
perf_df$norm_coverage <- normalize(perf_df$percent)

# Calculate performance gap
perf_df$perf_diff <- perf_df$norm_auc - perf_df$norm_coverage

# Define bar color
perf_df$color <- ifelse(perf_df$perf_diff > 0, "#55C667",  # green: classification better
                        ifelse(perf_df$perf_diff < 0, "#440154", "gray"))  # orange: regression better

# Plot
library(ggplot2)
ggplot(perf_df, aes(x = factor(property_id), y = perf_diff, fill = color)) +
  geom_bar(stat = "identity") +
  scale_fill_identity() +
  geom_hline(yintercept = 0, color = "black") +
  labs(
    title = NULL,
    y = "Relative Performance: Classification - Regression",
    x = "Municipality ID"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12), 
        axis.title.y = element_text(size = 12), 
        axis.title.x = element_text(size = 12))

ggsave(filename = "model_comparison1",
       plot = last_plot(),
       width = 300, # 14.1 x 5.05 in 358 x 256 mm 
       height = 200,# 
       units = "mm",
       dpi = 300,
       device = "pdf"
)
