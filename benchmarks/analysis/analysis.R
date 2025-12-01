library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# Read DIPD data
csv_list <- c(
  "../results/tsptw_rpid_t600_AFG_202511270953.csv", 
  "../results/tsptw_rpid_t600_GendreauDumasExtended_202511271054.csv", 
 # "../results/tsptw_rpid_t600_OhlmannThomas_202511271937.csv", 
  "../results/tsptw_rpid_t600_Solnon25_feasible_202511252213.csv", 
  "../results/tsptw_rpid_t600_Solnon25_infeasible_202511260948.csv", 
  "../results/tsptw_rpid_t600_SolomonPesant_202511280050.csv", 
  "../results/tsptw_rpid_t600_SolomonPotvinBengio_202511261103.csv"
)
tmp <- lapply(csv_list, read.csv)
didp_data <- do.call(rbind, tmp)
didp_data <- didp_data %>% mutate(Timeout=tolower(Timeout) == "true")
didp_data$Solver <- "DIDP"

# Read CODD data
csv_list <- c(
  "../results/tsptw_1_bro_t600_g_AFG_202511262302.csv",
  "../results/tsptw_1_bro_t600_g_GendreauDumasExtended_202511262343.csv", 
  #"../results/tsptw_1_bro_t600_g_OhlmannThomas_202511270542.csv", 
  "../results/tsptw_1_bro_t600_g_Solnon25_feasible_202511261450.csv", 
  "../results/tsptw_1_bro_t600_g_Solnon25_infeasible_202511262151.csv", 
  "../results/tsptw_1_bro_t600_g_SolomonPesant_202511262202.csv", 
  "../results/tsptw_1_bro_t600_g_SolomonPesant_202511280025.csv", 
  "../results/tsptw_1_bro_t600_g_SolomonPotvinBengio_202511262215.csv"
)
tmp <- lapply(csv_list, read.csv)
codd_data <- do.call(rbind, tmp)
codd_data <- codd_data %>% mutate(Timeout=tolower(Timeout) == "true")
codd_data$Solver <- "CODD"

csv_list <- c(
  "../results/tsptw_1_bro_t600_g_s_AFG_202511281915.csv",
  "../results/tsptw_1_bro_t600_g_s_GendreauDumasExtended_202511281957.csv",
  #"../results/tsptw_1_bro_t600_g_s_OhlmannThomas_202511290151.csv",
  "../results/tsptw_1_bro_t600_g_s_Solnon25_feasible_202511291349.csv",
  "../results/tsptw_1_bro_t600_g_s_Solnon25_infeasible_202511281755.csv",
  "../results/tsptw_1_bro_t600_g_s_SolomonPesant_202511281809.csv",
  "../results/tsptw_1_bro_t600_g_s_SolomonPotvinBengio_202511281833.csv"
)
tmp <- lapply(csv_list, read.csv)
codd_sorted_data <- do.call(rbind, tmp)
codd_sorted_data <- codd_sorted_data %>% mutate(Timeout=tolower(Timeout) == "true")
codd_sorted_data$Solver <- "CODDS"


# Sanity check
merged_data <- inner_join(didp_data, codd_data, by = c("Benchmark", "Instance"), suffix = c(".DIDP", ".CODD"))
mismatches <- merged_data %>% 
  filter(Best.Cost.DIDP != Best.Cost.CODD & 
        (!Timeout.DIDP & !Timeout.CODD))
if (nrow(mismatches) > 0){
  stop("❌ Mismatched Best values found:\n", paste(capture.output(print(mismatches)), collapse = "\n"))
}else{
  message("✅ All Best values match between solvers.")
}

#Normalization
merged_data$Benchmark[merged_data$Benchmark == "Solnon25_feasible"] <- "Solnon (Feasable)"
merged_data$Benchmark[merged_data$Benchmark == "Solnon25_infeasible"] <- "Solnon (Infeasible)"

# Analisys 1
# Solvedy by both, one solver > 1s and 
data_1 <- merged_data %>% 
  filter((Search.Time.DIDP >= 1 | Search.Time.CODD >= 1) & 
           (!Timeout.DIDP & !Timeout.CODD)) %>%
  mutate(
    Search.Time.Ratio = Search.Time.DIDP / Search.Time.CODD,
    Nodes.Ratio = Nodes.DIDP / Nodes.CODD
  )

summary_1 <- data_1 %>%
  group_by(Benchmark) %>%
  summarise(
    Count = n(),
    Avg.Search.Time.DIDP = mean(Search.Time.DIDP),
    Avg.Nodes.DIDP = mean(Nodes.DIDP),
    Avg.Search.Time.CODD = mean(Search.Time.CODD),
    Avg.Nodes.CODD = mean(Nodes.CODD),
    Avg.Search.Time.Ratio = mean(Search.Time.Ratio),
    Avg.Search.Time.Ratio2 = Avg.Search.Time.DIDP/Avg.Search.Time.CODD,
    Min.Search.Time.Ratio = min(Search.Time.Ratio),
    Max.Search.Time.Ratio = max(Search.Time.Ratio),
    StdDev.Search.Time.Ratio = sd(Search.Time.Ratio),
    Avg.Nodes.Ratio = mean(Nodes.Ratio),
    Min.Nodes.Ratio = min(Nodes.Ratio),
    Max.Nodes.Ratio = max(Nodes.Ratio),
    StdDev.Nodes.Ratio = sd(Nodes.Ratio)
  )

write.csv(summary_1, "summary_1.csv", row.names = FALSE)

# Analysis 2
# Prepare data - include all instances, assign timeout value to unsolved
timeout_value <- 600  # Set your actual timeout value
data_2 <- merged_data %>%
  pivot_longer(
    cols = c(Search.Time.DIDP, Search.Time.CODD),
    names_to = "Solver",
    values_to = "Search.Time"
  ) %>%
  mutate(
    Solver = ifelse(grepl("DIDP", Solver), "DIDP", "CODD"),
    Timeout = ifelse(Solver == "DIDP", Timeout.DIDP, Timeout.CODD),
    # Use timeout value for unsolved instances
    Search.Time = ifelse(Timeout, timeout_value*2, Search.Time)
  )

# Plot with stat_ecdf
# Reverse the factor order so CODD (blue) is drawn last (on top)
data_2$Solver <- factor(data_2$Solver, levels = c("DIDP", "CODD"))

# Plot with stat_ecdf
ggplot(data_2, aes(x = Search.Time, color = Solver)) +
  stat_ecdf(geom = "step", linewidth = 0.5, pad = TRUE) +
  facet_wrap(~ Benchmark) +
  coord_cartesian(xlim = c(0, timeout_value)) +
  scale_y_continuous(limits = c(0.4, 1), labels = scales::label_percent(suffix = "")) +
  scale_color_manual(
    values = c( DIDP = "#CC6F3C", CODD = "#76B900"),  
    labels = c("DIDP-Rust","CODD-GPU")) +
  labs(x = "Time [s]", 
       y = "Solved Instances [%]",
       color = "Solver") +
  theme_minimal() +
  theme(
    legend.position = c(0.2, 0.68),
    legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.key.size = unit(0.35, "cm"),
    legend.spacing.y = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0),
    panel.grid.minor = element_blank()
  )
ggsave("plot2.pdf", width = 8, height = 4, units = "in", dpi = 300)

# Analysis 3 
data_3 <- merged_data %>% filter(grepl("Solnon", Benchmark)) 
data_3 <- data_3 %>% filter(grepl("Feas", Benchmark)) 
data_3 <- data_3 %>% filter(
  (Search.Time.DIDP >= 3 & Search.Time.CODD >= 3) & 
  (!Timeout.DIDP | !Timeout.CODD)) # Important! We assume the lower cost as opt
data_3 <- data_3 %>%
  mutate(
    Optimal.Cost = pmin(Best.Cost.DIDP, Best.Cost.CODD, na.rm = TRUE)
  )
data_3$Search.Time.Ratio <- data_3$Search.Time.DIDP / data_3$Search.Time.CODD

# Prepare data for plot
data_3_didp <- data_3 %>% transmute(
  Instance,
  Optimal.Cost = Optimal.Cost,
  Solver = "DIDP",
  Best.Cost = Best.Cost.DIDP,
  Best.Time = Best.Time.DIDP,
  Search.Time = Search.Time.DIDP ) 
data_3_codd <- data_3 %>% transmute( 
  Instance,
  Optimal.Cost = Optimal.Cost,
  Solver = "CODD",
  Best.Cost = Best.Cost.CODD,
  Best.Time = Best.Time.CODD, 
  Search.Time = Search.Time.CODD ) 
data_3_plot <- bind_rows(data_3_didp, data_3_codd)
data_3_plot <- data_3_plot %>%
  mutate(Best.Time = ifelse(
    is.na(Best.Cost) | Best.Cost != Optimal.Cost, 
    Search.Time, 
    Best.Time
  ))
data_3_plot$Delta.Time <- data_3_plot$Search.Time - data_3_plot$Best.Time

palette <- c(
  # DIDP.Opt = "#D55E00",      # Vermillion/orange-red
  # DIDP.Proof = "#F0A030",     # Light orange
  DIDP.Opt   = "#CC6F3C",   # Deeper Rust crab tone
  DIDP.Proof = "#F2A67A",   # Matching lighter shade
  
  CODD.Opt   = "#76B900",   # Official Nvidia Green
  CODD.Proof = "#A8E65C"   # Light Nvidia Green (derived)
  
  # CODD.Opt = "#0072B2",      # Strong blue
  # CODD.Proof = "#56B4E9"    # Light blue
)
# Reshape data for proper stacking
data_3_long <- data_3_plot %>%
  select(Instance, Solver, Best.Time, Delta.Time) %>%
  pivot_longer(
    cols = c(Best.Time, Delta.Time),
    names_to = "Component",
    values_to = "Time"
  ) %>%
  mutate(
    Combined = case_when(
      Solver == "DIDP" & Component == "Best.Time" ~ "DIDP.Opt",
      Solver == "DIDP" & Component == "Delta.Time" ~ "DIDP.Proof",
      Solver == "CODD" & Component == "Best.Time" ~ "CODD.Opt",
      Solver == "CODD" & Component == "Delta.Time" ~ "CODD.Proof"
    ),
    Combined = factor(
      Combined,
      levels = c("DIDP.Opt", "DIDP.Proof", "CODD.Opt", "CODD.Proof")
    )
  )
ggplot(data_3_long, aes(x = Instance, y = Time, fill = Combined)) +
  geom_bar(stat = "identity", position = position_stack(reverse = TRUE)) +
  facet_wrap(~ Solver, scales = "free_x") +
  scale_fill_manual(
      values = palette,
      labels = c(
        "DIDP-Rust Opt", "DIDP-Rust Proof",
        "CODD-GPU Opt", "CODD-GPU Proof")
    ) +
  labs(
    x = "Instance",
    y = "Time [s]"
  ) +
    theme_minimal() +
    theme(
      strip.text = element_blank(),
      axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.5, size = 8),
      axis.text.y = element_text(size = 9),
      axis.title.y = element_text(size = 10, margin = margin(r = 5)),
      legend.position = c(0.12, 0.65),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.35, "cm"),
      legend.spacing.y = unit(0, "pt"),
      legend.title = element_blank(),
      plot.margin = margin(0,0,0,0),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

ggsave("plot3.pdf", width = 9, height = 3, units = "in", dpi = 300)

# Analisys 4
# Sanity check
merged_data2 <- inner_join(codd_sorted_data, codd_data, by = c("Benchmark", "Instance"), suffix = c(".CODDS", ".CODD"))

#Normalization
merged_data2$Benchmark[merged_data2$Benchmark == "Solnon25_feasible"] <- "Solnon (Feasable)"
merged_data2$Benchmark[merged_data2$Benchmark == "Solnon25_infeasible"] <- "Solnon (Infeasible)"


# Solvedy by both, one solver > 1s and 
data_4 <- merged_data2 %>% 
  filter((Search.Time.CODDS >= 1 | Search.Time.CODD >= 1) &
           (!Timeout.CODDS & !Timeout.CODD)) %>%
  mutate(
    Search.Time.Ratio = Search.Time.CODDS / Search.Time.CODD,
    Nodes.Ratio = Nodes.CODDS / Nodes.CODD
  )

data_4_better <- merged_data2 %>% 
  filter(
           (Timeout.CODDS != Timeout.CODD))

summary_4 <- data_4 %>%
  group_by(Benchmark) %>%
  summarise(
    Count = n(),
    Avg.Search.Time.CODDS = mean(Search.Time.CODDS),
    Avg.Nodes.CODDS = mean(Nodes.CODDS),
    Avg.Search.Time.CODD = mean(Search.Time.CODD),
    Avg.Nodes.CODD = mean(Nodes.CODD),
    Avg.Search.Time.Ratio = mean(Search.Time.Ratio),
    Min.Search.Time.Ratio = min(Search.Time.Ratio),
    Max.Search.Time.Ratio = max(Search.Time.Ratio),
    StdDev.Search.Time.Ratio = sd(Search.Time.Ratio),
    Avg.Nodes.Ratio = mean(Nodes.Ratio),
    Min.Nodes.Ratio = min(Nodes.Ratio),
    Max.Nodes.Ratio = max(Nodes.Ratio),
    StdDev.Nodes.Ratio = sd(Nodes.Ratio)
  )

write.csv(summary_4, "summary_4.csv", row.names = FALSE)



