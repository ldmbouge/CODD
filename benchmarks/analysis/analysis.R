library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)


# Read DIPD data
didp_data <- read.csv("./didp_tsptw_1800.csv")
didp_data$Solver <- "DIDP"

# Reads CODD data
#codd_data_afg <-   read.csv("./tsptw_gpu_1_cbs_g_600_AFG_202510141257.csv")
#codd_data_dumas <- read.csv("./tsptw_gpu_1_cbs_g_600_Dumas_202510141343.csv")
codd_data_solnon_f <- read.csv("./tsptw_gpu_1_cbs_g_1800_Solnon25_feasible_202510162003_slim.csv")
codd_data_solnon_i <- read.csv("./tsptw_gpu_1_cbs_g_1800_Solnon25_infeasible_202510171353_slim.csv")
codd_data <- rbind(
  #codd_data_afg, codd_data_dumas,
  codd_data_solnon_f,codd_data_solnon_i)
codd_data$Solver <- "CODD"

# Results check
merged_data <- inner_join(didp_data, codd_data, by = c("Benchmark", "Instance"), suffix = c("DIDP", "CODD"))
mismatches <- merged_data %>% filter(BestDIDP != BestCODD)
print(mismatches)
if (nrow(mismatches) > 0){
  stop("❌ Mismatched Best values found:\n", paste(capture.output(print(mismatches)), collapse = "\n"))
}else{
  message("✅ All Best values match between solvers.")
}

# Speedup
more_than_1 <- merged_data[merged_data$Proof.TimeDIDP > 1, ]
more_than_1$Speedup <- more_than_1$Proof.TimeDIDP / more_than_1$Proof.TimeCODD
more_than_1_clean <- more_than_1 %>%
  filter(!is.na(Benchmark), !is.na(Proof.TimeCODD), !is.na(Proof.TimeDIDP))
write.csv(more_than_1_clean, "moreThen1Sec.csv", row.names = FALSE)

#Plot
plot_data <- more_than_1 %>%
  pivot_longer(
    cols = c(Proof.TimeDIDP, Proof.TimeCODD),
    names_to = "Solver",
    values_to = "ProofTime"
  )
plot_data <- plot_data %>%
  filter(!is.na(Benchmark), !is.na(Instance), !is.na(ProofTime))
sum(is.na(plot_data$Benchmark))

ggplot(plot_data, aes(x = Instance, y = ProofTime, fill = Solver)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Benchmark, scales = "free") +
  scale_y_continuous(
    trans = pseudo_log_trans(base = 10, sigma = 1e-1),  # sigma defines the smooth area near 0
  name = "Proof Time (pseudo-log scale)"
  ) +
  labs(
    title = "Proof Time Comparison (DIDP vs CODD)",
    x = "Instance",
    y = "Proof Time (s)",
    fill = "Solver"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )

