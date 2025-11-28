library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# Read DIPD data
csv_list <- c(
  "../results/tsptw_rpid_t600_Solnon25_feasible_202511252213.csv", 
  "../results/tsptw_rpid_t600_Solnon25_infeasible_202511260948.csv"
)
tmp <- lapply(csv_list, read.csv)
didp_data <- do.call(rbind, tmp)
didp_data <- didp_data %>% mutate(Timeout=tolower(Timeout) == "true")
didp_data$Solver <- "DIDP"

# Read CODD data
csv_list <- c(
  "../results/tsptw_1_bro_t600_g_Solnon25_feasible_202511261450.csv", 
  "../results/tsptw_1_bro_t600_g_Solnon25_infeasible_202511262151.csv"
)
tmp <- lapply(csv_list, read.csv)
codd_data <- do.call(rbind, tmp)
codd_data <- codd_data %>% mutate(Timeout=tolower(Timeout) == "true")
codd_data$Solver <- "CODD"

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

# Analisys 1
# Solvedy by both, one solver > 1s and 
merged_data_1 <- merged_data %>% 
  filter((Search.Time.DIDP >= 1 | Search.Time.CODD >= 1) & 
         (!Timeout.DIDP & !Timeout.CODD))
summary_1 <- merged_data_1 %>%
  group_by(Benchmark) %>%
  summarise(
    Avg.Search.Time.DIDP = mean(Search.Time.DIDP),
    Avg.Nodes.DIDP = mean(Nodes.DIDP),
    Avg.Search.Time.CODD = mean(Search.Time.CODD),
    Avg.Nodes.CODD = mean(Nodes.CODD)
  )
summary_1$Avg.Search.Time.Ratio <- summary_1$Avg.Search.Time.DIDP / summary_1$Avg.Search.Time.CODD
summary_1$Avg.Nodes.Ratio <- summary_1$Avg.Nodes.DIDP / summary_1$Avg.Nodes.CODD
write.csv(summary_1, "summary_1.csv", row.names = FALSE)


# 1. Prepare solver specific tables
# Analysis 3 
data_3 <- merged_data %>% filter(grepl("Solnon25_feas", Benchmark)) 
data_3 <- data_3 %>% filter(
  (Search.Time.DIDP >= 3 & Search.Time.CODD >= 3) & 
  (!Timeout.DIDP | !Timeout.CODD)) 
data_3$Search.Time.Ratio <- data_3$Search.Time.DIDP / data_3$Search.Time.CODD 
data_3$Nodes.Ratio <- data_3$Nodes.DIDP / data_3$Nodes.CODD 

# Prepare data for plot
data_3_didp <- data_3 %>% transmute(
  Instance,
  Solver = "DIDP",
  Best.Time = Best.Time.DIDP,
  Search.Time = Search.Time.DIDP ) 
data_3_codd <- data_3 %>% transmute( 
  Instance,
  Solver = "CODD", 
  Best.Time = Best.Time.CODD, 
  Search.Time = Search.Time.CODD ) 
data_3_plot <- bind_rows(data_3_didp, data_3_codd)
data_3_plot$Delta.Time <- ifelse(
  is.na(data_3_plot$Best.Time),
  0, 
  data_3_plot$Search.Time - data_3_plot$Best.Time ) 
data_3_plot$Best.Time <- ifelse(
  is.na(data_3_plot$Best.Time), 
  data_3_plot$Search.Time, 
  data_3_plot$Best.Time ) 
# Define palette with meaningful mapping
# palette <- c(
#   CODD.Best = "#76B900",
#   CODD.Search = "#97EC00",
#   DIDP.Best = "#F2800A",
#   DIDP.Search = "#EFA94D"
# )
palette <- c(
  CODD.Best = "#0072B2",      # Strong blue
  CODD.Search = "#56B4E9",    # Light blue
  DIDP.Best = "#D55E00",      # Vermillion/orange-red
  DIDP.Search = "#F0A030"     # Light orange
)
# Reshape data for proper stacking
data_3_stacked <- data_3_plot %>%
  mutate(Solver_Best = paste0(Solver, ".Best"),
         Solver_Search = paste0(Solver, ".Search")) %>%
  select(Instance, Solver, Best.Time, Delta.Time, Solver_Best, Solver_Search)
timeout_value <- 600  # Adjust to your actual timeout value
ggplot(data_3_stacked, aes(x = Instance)) +
  geom_col(aes(y = Delta.Time, fill = Solver_Search), 
           position = position_dodge(width = 0.8), width = 0.7) +
  geom_col(aes(y = Best.Time, fill = Solver_Best), 
           position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = timeout_value, 
             color = "red", 
             linewidth = 0.8) +
  annotate("text", x = 1, y = timeout_value, 
           label = "Timeout", 
           vjust = -0.5, 
           hjust = 0, 
           size = 3) +
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5))) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = palette,
                    labels = c("CODD-GPU Best", "CODD-GPU Search", 
                               "DIDP-Rust Best", "DIDP-Rust Search")) +
  labs(x = "Instance", y = "Time [s]", fill = NULL) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.5, size = 8),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(size = 10, margin = margin(r = 5)),
    legend.position = c(0.1, 0.6),
    legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
    legend.text = element_text(size = 7),
    legend.key.size = unit(0.35, "cm"),
    legend.spacing.y = unit(0, "pt"),
    plot.margin = margin(4, 2, 2, 2),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  coord_cartesian(clip = "off") 
ggsave("plot3.pdf", width = 9, height = 3, units = "in", dpi = 300)
